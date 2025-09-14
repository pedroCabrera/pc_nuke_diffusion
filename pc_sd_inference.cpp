#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"
#include "DDImage/Executable.h"
#include <memory>

// Include Stable Diffusion headers
#include "stable-diffusion.h"
#include "stable_diffusion_wrapper.h"

using namespace DD::Image;

static const char* const CLASS = "pc_sd_inference";
static const char* const HELP = "Stable Diffusion For Nuke";

class pc_sd_inference : public Iop, Executable{
    std::vector<const char*> schedule_names;
    std::vector<const char*>sample_method_names;
    const char* prompt = "A Cgi render of a cute corgi";
    const char* negative_prompt = "Ugly, blurry, low resolution";

    int seed = -1;
    int sample_method = 0;    
    int schedule = 0;
    bool vae_tiling = false;  
    double format_w, format_y;
    int cached_format_w, cached_format_y;
    float strength = 1.0;
    int sample_steps = 20;
    int clip_skip = -1;
    float cfg_scale = 7.5;
    float control_strength = 20.0;
    float style_strength = 20.0;
    bool normalize_input = false;
    float min_cfg = 1.0;
    bool canny_preprocess = false;

    bool temp_save = false;
    const char* temp_path = "[python -execlocal {import tempfile;ret = tempfile.gettempdir()+'/pc_nuke_diffusion'}]";

    bool use_init_image_as_ref = false;
    sd_image_t control_image; 
    sd_image_t* results = NULL;
    sd_image_t init_image;
    sd_image_t mask_image;
    std::vector<sd_image_t> ref_images;
    Format *customFormat;
    Knob* __formatknob;
    
    int token_ = 0;

    virtual Executable* executable() { return this; }



public:
    pc_sd_inference(Node* node) : Iop(node), Executable(this) {
        inputs(7);
        format_w = format_y = 512;   
        schedule_names = get_schedule_names();    
        sample_method_names = get_sample_method_names();
    }
    bool debug_mode = true;
    std::atomic<unsigned> m_progress{0};
    std::string m_progress_message = "Inferencing...";
    sd_ctx_t* sd_ctx = NULL;

    const char* input_label(int input, char* buffer) const
    {
    switch (input) {
        case 0:
        return "Input Image";        
        case 1:
        return "Model";
        case 2:
        return "ControlNet Image";  
        case 3:
        return "Ref_1";              
        case 4:
        return "Ref_2";   
        case 5:
        return "Ref_3";
        case 6:
        return "Ref_4";                        
        default:
        return 0;
    }
    }
    //int minimum_inputs() const { return 1; }
    int optional_input() const {return 3;};
    
    void beginExecuting(){}
    void endExecuting(){}  
    void execute();
    
    bool isConnected(int inputIndex) {
        return (node_input(inputIndex)!=nullptr);
    }

    void _validate(bool for_real)
    {
        if(!isConnected(0)) {
        delete customFormat;
        customFormat = new Format(int(format_w), int(format_y));
        info_.format(*customFormat);
        info_.full_size_format(*customFormat);
        info_.x(0);
        info_.y(0);
        info_.r(int(format_w));
        info_.t(int(format_y));
        info_.turn_on(Mask_RGB);
        } else {
            copy_info(0);
            info_.turn_on(Mask_RGB);
            delete customFormat;
            Format inpFormat = input(0)->format();
            customFormat = new Format(int(inpFormat.w()), int(inpFormat.h()));
            if (__formatknob) __formatknob->set_value(inpFormat.w(),0);
            if (__formatknob) __formatknob->set_value(inpFormat.h(),1);
            format_w = inpFormat.w();
            format_y = inpFormat.h();
        }
        if(cached_format_w != format_w || cached_format_y != format_y){
            printf("Format changed, clearing cache\n");
            free(results);
            results = NULL;
            cached_format_w = format_w;
            cached_format_y = format_y;
        }        
    }

    void getRequests(const Box& bbox, const DD::Image::ChannelSet& channels, int count, RequestOutput &reqData) const
    {
        reqData.request( input(0), bbox, channels + Chan_Alpha, count);
        reqData.request( input(3), bbox, channels, count);
        reqData.request( input(4), bbox, channels, count);
        reqData.request( input(5), bbox, channels, count);
        reqData.request( input(6), bbox, channels, count);
        //reqData.request( input(2), bbox, channels, count);
    }    
    
    unsigned int getFrameProgress() const override {            // 0..100
        return m_progress.load(std::memory_order_relaxed);
    }
    std::string getFrameProgressMessage() const override {
        return m_progress_message;
    }


    void sample_sd(){
        // Get the input node
        if (!isConnected(1)) {
            error("Connect a model.");
            return;
        }
        // Check if the input node implements AbstractDataInterface
        AbstractDataInterface* dataNode = dynamic_cast<AbstractDataInterface*>(node_input(1));
        if (!dataNode) {
            error("Connect a pc_sd_load_model Node please.");
            return;
        }

        // Access the shared data
        sd_ctx = dataNode->get_ctxt();
        
        if (sd_ctx) {
            sd_image_t sample_init_image = {(uint32_t)format_w, (uint32_t)format_y, 3, NULL};
            if(isConnected(0)) {
                printf("Input Image connected\n"); 
                sd_images_out init_data = input2sdimages(input(0),format_w,format_y,true);
                init_image = init_data.rgb;
                mask_image = init_data.alpha;
                if(init_data.has_alpha){
                    printf("Image has a mask\n"); 
                }
                if(use_init_image_as_ref){
                    ref_images.push_back(init_image);
                }
                if(strength < 1.0){
                    sample_init_image = init_image;                    
                }
            }
            for (size_t i = 3; i < 7; i++)
            {
                if(isConnected(i)){
                    printf("Ref Image %zu connected\n",i-2); 
                    sd_images_out init_data = input2sdimages(input(i),format_w,format_y,false);
                    ref_images.push_back(init_data.rgb);            
                };
            }

            sd_sample_params_t sample_params;
            sd_sample_params_init(&sample_params);

            sample_params.scheduler = (scheduler_t)schedule;
            sample_params.sample_method = (sample_method_t)sample_method;
            sample_params.sample_steps = sample_steps;

            sample_params.guidance.txt_cfg = cfg_scale;
            sample_params.guidance.img_cfg = cfg_scale;
            //sample_params.guidance.distilled_guidance = cfg_scale;

            sd_img_gen_params_t img_gen_params;
            sd_img_gen_params_init(&img_gen_params);

            img_gen_params.prompt = prompt;
            img_gen_params.negative_prompt = negative_prompt;
            img_gen_params.clip_skip = clip_skip;
            img_gen_params.init_image = sample_init_image;
            img_gen_params.ref_images = ref_images.data();
            img_gen_params.ref_images_count = (int)ref_images.size();
            img_gen_params.increase_ref_index = false;//params.increase_ref_index;
            img_gen_params.mask_image = mask_image;
            img_gen_params.width = format_w;
            img_gen_params.height = format_y;
            img_gen_params.sample_params = sample_params;
            img_gen_params.strength = strength;
            img_gen_params.seed = seed;
            img_gen_params.batch_count = 1;//params.batch_count,
            img_gen_params.control_image = control_image;
            img_gen_params.control_strength = control_strength;
            img_gen_params.style_strength = style_strength;
            img_gen_params.normalize_input = normalize_input;
            img_gen_params.input_id_images_path = "";//params.input_id_images_path.c_str(),

            printf("inferencing\n");
            free(results);
            results     = generate_image(sd_ctx, &img_gen_params);
            cached_format_w = format_w;
            cached_format_y = format_y;

            if(isConnected(1)){
                if(use_init_image_as_ref == false){
                    printf("Cleaning Input IMAGE\n");
                    free(init_image.data);
                }
                printf("Cleaning Mask\n");
                free(mask_image.data); 
            }
            printf("Cleaning Refs\n");
            for (auto ref_image : ref_images) {
                if(ref_image.data ){
                    free(ref_image.data);
                    ref_image.data = NULL;
                }
            }
            ref_images.clear();

            if (results == NULL) {
                error("inference failed\n");
                return;
            } else{
                printf("inference done\n");
                if(temp_save){
                    save_image(results[0], temp_path, false);
                    printf("Saved temporal files to %s\n", temp_path);
                }
            }            
        } else {
            error("Model is not loaded.");
            return;
        }
    }
    
    void engine(int y, int x, int r, ChannelMask channels, Row& out) override {
        out.erase(channels);                  // zero requested channels first

        if (results == NULL) return;

        const Format f = format();            // current output format
        const int fx = f.x(), fy = f.y(), fr = f.r(), ft = f.t();
        const int W  = fr - fx;
        const int H  = ft - fy;

        // clip to our bounds (defensive: engine can be called with larger x..r)
        int xi0 = (std::max)(x, fx);
        int xi1 = (std::min)(r, fr);
        if (xi0 >= xi1 || y < fy || y >= ft) return;

        // If your cached image has top-left origin, flip Y like your Iop code:
        const bool flipY = true;
        const int srcY   = flipY ? (H - 1 - (y - fy)) : (y - fy);

        // base index into your packed RGB list
        size_t idx = (size_t(srcY) * W + (xi0 - fx)) * 3;

        // Grab writable channel pointers (only if requested)
        float* R = (channels & Mask_Red)   ? out.writable(Chan_Red)   + xi0 : nullptr;
        float* G = (channels & Mask_Green) ? out.writable(Chan_Green) + xi0 : nullptr;
        float* B = (channels & Mask_Blue)  ? out.writable(Chan_Blue)  + xi0 : nullptr;

        for (int xi = xi0; xi < xi1; ++xi) {
            float r8 = static_cast<float>(results[0].data[idx++]);
            float g8 = static_cast<float>(results[0].data[idx++]);
            float b8 = static_cast<float>(results[0].data[idx++]);

            // match your original scale (you had /10); change to /255.f if you want normalized
            if (R) *R++ = r8 / 255.f;
            if (G) *G++ = g8 / 255.f;
            if (B) *B++ = b8 / 255.f;
        }
    }

    void knobs(Knob_Callback f)
    {
        const char* renderScript = "nuke.execute(nuke.thisNode(), nuke.frame(), nuke.frame(), 1)";
        PyScript_knob(f, renderScript, "Execute");

        Bool_knob(f, &temp_save, "save_temporal_files", "save_temporal_files");
        SetFlags(f, Knob::STARTLINE );  

        File_knob(f, &temp_path, "temp_path", "temp_path");
        SetFlags(f, Knob::STARTLINE );

        Int_knob(f, &token_, "token", "");               // hidden version knob
        SetFlags(f, Knob::INVISIBLE | Knob::NO_ANIMATION);
        Bool_knob(f, &use_init_image_as_ref, "use_init_image_as_ref", "use_init_image_as_ref");
        SetFlags(f, Knob::STARTLINE );

        __formatknob = WH_knob(f,&format_w,"format","format");
        SetFlags(f, Knob::SLIDER | Knob::NO_PROXYSCALE);
        ClearFlags(f, Knob::MAGNITUDE);
    
        Multiline_String_knob(f, &prompt, "prompt", "prompt",5 );
        Multiline_String_knob(f, &negative_prompt, "negative_prompt", "negative_prompt", 5 );
        
        Int_knob(f, &seed, "seed", "seed");
        SetFlags(f, Knob::SLIDER);
        //SetRange(f, IRange(0, 100000));
        //Float_knob(f, &min_cfg, "min_cfg", "min_cfg");
        Float_knob(f, &cfg_scale, "cfg_scale", "cfg_scale");
        Float_knob(f, &strength, "strength", "strength");
        Bool_knob(f, &normalize_input, "normalize_input", "normalize_input");
        //Float_knob(f, &control_strength, "control_strength", "control_strength");
        //Float_knob(f, &style_strength, "style_strength", "style_strength");
        Int_knob(f, &sample_steps, "sample_steps", "sample_steps");
        SetFlags(f, Knob::SLIDER); 
        Enumeration_knob(f, &sample_method, sample_method_names.data(), "sample_method", "sample_method");   
        Enumeration_knob(f, &schedule, schedule_names.data(), "schedule", "schedule");  
        //Bool_knob(f, &vae_tiling, "vae_tiling", "vae_tiling");
        Int_knob(f, &clip_skip, "clip_skip", "clip_skip");
        //Bool_knob(f,&canny_preprocess, "canny_preprocess","canny_preprocess");

    }    
    const char* Class() const { return desc.name; };
    const char* node_help() const { return HELP; };

    static const Iop::Description desc;  
};

void sd_prog_call(int step, int steps, float time, void* data){
    unsigned p = static_cast<unsigned>((step) * 100 / steps);
    pc_sd_inference* node = (pc_sd_inference*)data;
    node->m_progress.store(p, std::memory_order_relaxed);
    char buf[64];
    snprintf(buf, sizeof(buf), "inferencing... %d/%d - %.2f s/it",step,steps, time);    
    node->m_progress_message = buf;
    nuke_pretty_progress(step, steps, time);
    //if( node->aborted() ){
    //    if(node->sd_ctx) sd_request_cancel(node->sd_ctx);
    //    printf("User aborted\n");
    //}
}
void pc_sd_inference::execute() {
    sd_set_progress_callback(sd_prog_call, this);
    m_progress.store(0, std::memory_order_relaxed);
    m_progress_message = "Starting...";
    sample_sd();
    ++token_;
    Knob* t = knob("token");
    if (t) t->set_value(token_);        
}
static Iop* pc_sd_inferenceCreate(Node* node)
{
  return new pc_sd_inference(node);
}
const Iop::Description pc_sd_inference::desc(CLASS, "pc_nuke_diffusion", pc_sd_inferenceCreate);
