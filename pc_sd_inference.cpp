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

    int mode = IMG_GEN;
    int video_mode = IMG2VID;
    int output_index = 0;

    std::vector<const char*> schedule_names;
    std::vector<const char*>sample_method_names;

    int sample_method = 0; 
    int schedule = 0;
    bool vae_tiling = false;  
    int tile_size = 32;
    float tile_overlap = 0.5f;
    // Format
    double format_w, format_y;
    int cached_format_w, cached_format_y;
    Format *customFormat;
    Knob* __formatknob;
    // Image Generation
    const char* prompt =    "Black and white dog, sitting, shaggy fur, cartoon style, expressive, whimsical,  medium shot,"
                            "detailed fur texture,  unconventional hairstyle,  light background,  studio lighting,"
                             "high detail, 8k,  vibrant colors,  intricate details,  dramatic lighting,  cinematic composition,"
                             "soft lighting,  bright colors,  highly detailed,  sharp focus,  award winning photography,"
                             "masterpiece,  best quality,  ultra-detailed,  photo-realistic,  realistic texture,"
                             "intricate patterns,  symmetrical design,  ornate details,  elegant composition,"
                             "beautiful lighting,  vibrant colors,  rich textures,  captivating details,"
                             "fantasy art,  surreal landscape,  magical atmosphere,  ethereal quality,"
                             "dynamic pose,  expressive face,";
                         ;
    const char* negative_prompt =   "Ugly, blurry, low resolution, deformed, disfigured, mutated, extra limbs,"
                                    "poorly drawn, bad anatomy, blurry, fuzzy, out of focus, long neck, long body,"
                                    "mutated hands and fingers, poorly drawn hands and fingers, missing fingers,"
                                    "extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, ugly";
    int seed = -1;
    float strength = 1.0;
    int sample_steps = 20;
    int high_noise_sample_steps = -1;


    int clip_skip = -1;
    float txt_cfg = 7.5;
    float img_cfg = -1.0;
    float guidance = 3.5;

    float high_noise_txt_cfg = 7.0;
    float high_noise_img_cfg = -1.0;
    float high_noise_guidance = 3.5;

    float slg_scale = 0.0f;
    float slg_layer_start = 0.01f;
    float slg_layer_end = 0.2f;

    float high_noise_slg_scale = 0.0f;
    float high_noise_slg_layer_start = 0.01f;
    float high_noise_slg_layer_end = 0.2f;

    // ControlNet
    float control_strength = 0.9;
    // Photo Maker
    float pm_style_strength = 20.0;
    const char* pm_id_embed_path = "";
    // Video Generation
    int curr_frame = 1;
    int frame_range[2] = {1,7}; // start, end
    float moe_boundary = 0.875f;
    float vace_strength = 1.f;
    
    // Temporal Files
    bool temp_save = false;
    const char* temp_path = "[python -execlocal {import tempfile;ret = tempfile.gettempdir()+'/pc_nuke_diffusion'}]";

    bool use_init_image_as_ref = false;

    sd_image_t* results = NULL;
    int num_results = 1;
    sd_image_t init_image;
    sd_image_t mask_image;
    sd_image_t end_image;
    sd_image_t control_image;

    std::vector<sd_image_t> ref_images;
    std::vector<sd_image_t> control_frames;
    std::vector<sd_image_t> pmid_images;
    
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    sd_img_gen_params_t img_gen_params;
    sd_vid_gen_params_t vid_gen_params;
    sd_tiling_params_t vae_tiling_params = {false, 0, 0, 0.5f, 0.0f, 0.0f};
    
    int token_ = 0;

    virtual Executable* executable() { return this; }



public:
    pc_sd_inference(Node* node) : Iop(node), Executable(this) {
        inputs(7);
        format_w = format_y = 512;   
        schedule_names = get_schedule_names();    
        sample_method_names = get_sample_method_names();
        sd_sample_params_init(&sample_params);
        sd_img_gen_params_init(&img_gen_params);  
        sd_sample_params_init(&high_noise_sample_params);
        sd_vid_gen_params_init(&vid_gen_params);
        high_noise_sample_params.sample_steps = -1;              
    }
    bool debug_mode = true;
    std::atomic<unsigned> m_progress{0};
    std::string m_progress_message = "Inferencing...";
    sd_ctx_t* sd_ctx = NULL;

    const char* input_label(int input, char* buffer) const{
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
    
    // Referenced inputs are optional
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
            if (results != NULL) {
            for (int i = 0; i < num_results; i++) {
                free(results[i].data);
                results[i].data = NULL;
            }}
            free(results);
            results = NULL;
            cached_format_w = format_w;
            cached_format_y = format_y;
        }        
    }

    void getRequests(const Box& bbox, const DD::Image::ChannelSet& channels, int count, RequestOutput &reqData) const
    {
        reqData.request( input(0), bbox, channels + Chan_Alpha, count);
        reqData.request( input(2), bbox, channels, count);
        reqData.request( input(3), bbox, channels, count);
        reqData.request( input(4), bbox, channels, count);
        reqData.request( input(5), bbox, channels, count);
        reqData.request( input(6), bbox, channels, count);
    }    
    
    unsigned int getFrameProgress() const override {
        return m_progress.load(std::memory_order_relaxed);
    }

    std::string getFrameProgressMessage() const override {
        return m_progress_message;
    }


    void sample_sd(){
        // Check if the model input is connected
        if (!isConnected(1)) {
            error("Connect a model.");
            return;
        }
        // Check if the input node implements model_loaderNode
        model_loaderNode* dataNode = dynamic_cast<model_loaderNode*>(node_input(1));
        if (!dataNode) {
            error("Connect a pc_sd_load_model Node please.");
            return;
        }

        // Access the shared data
        sd_ctx = dataNode->get_ctxt();
        
        if (sd_ctx) {
            // Scheduler and Sample params
            sample_params.scheduler = (scheduler_t)schedule;
            sample_params.sample_method = (sample_method_t)sample_method;
            sample_params.sample_steps = sample_steps;
            sample_params.guidance.txt_cfg = txt_cfg;
            sample_params.guidance.img_cfg = img_cfg;
            sample_params.guidance.distilled_guidance = guidance;
            sample_params.guidance.slg.scale = slg_scale;
            sample_params.guidance.slg.layer_start = slg_layer_start;
            sample_params.guidance.slg.layer_end = slg_layer_end;

            high_noise_sample_params.sample_steps = high_noise_sample_steps;

            if (sample_params.guidance.img_cfg < 0.0f) {
                sample_params.guidance.img_cfg = sample_params.guidance.txt_cfg;
            }

            if (high_noise_sample_params.guidance.img_cfg < 0.0f) {
                high_noise_sample_params.guidance.img_cfg = high_noise_sample_params.guidance.txt_cfg;
            }

            vae_tiling_params.enabled = vae_tiling;
            vae_tiling_params.tile_size_x = tile_size;
            vae_tiling_params.tile_size_y = tile_size;
            vae_tiling_params.target_overlap = tile_overlap;

            printf("Cleaning Cached Results\n");
            if (results != NULL) {
                for (int i = 0; i < num_results; i++) {
                    free(results[i].data);
                    results[i].data = NULL;
                }}
            free(results);
            results = NULL;

            cached_format_w = format_w;
            cached_format_y = format_y;
            num_results = 1;
            sd_image_t sample_init_image = {(uint32_t)format_w, (uint32_t)format_y, 3, NULL};

            // Input Image
            if(isConnected(0)) {
                printf("Input Image connected\n"); 
                sd_images_out init_data = input2sdimages(input(0),(int)format_w,(int)format_y,true);
                init_image = init_data.rgb;
                mask_image = init_data.alpha;
                if(init_data.has_alpha){
                    printf("Image has a mask\n"); 
                }
                if(use_init_image_as_ref){
                    ref_images.push_back(init_image);
                }
                sample_init_image = init_image;
            }

            if (mode == IMG_GEN){
                // Control Net Image
                if(isConnected(2)) {
                    printf("Control Net Image connected\n"); 
                    sd_images_out init_data = input2sdimages(input(2),(int)format_w,(int)format_y,false);
                    control_image = init_data.rgb;
                }
                // Reference Images
                for (size_t i = 3; i < 7; i++)
                {
                    if(isConnected(i)){
                        printf("Ref Image %zu connected\n",i-2);
                        sd_images_out init_data = input2sdimages(input(i),(int)format_w,(int)format_y,false);
                        ref_images.push_back(init_data.rgb);
                    };
                }

                // Image Params
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
                img_gen_params.pm_params = {
                    pmid_images.data(),
                    (int)pmid_images.size(),
                    pm_id_embed_path,
                    pm_style_strength,
                },                  
                img_gen_params.vae_tiling_params = vae_tiling_params;     
                printf("Generating Image\n");
                results     = generate_image(sd_ctx, &img_gen_params);
            }else if (mode == VID_GEN){

                num_results = (frame_range[1] - frame_range[0]);

                vid_gen_params.prompt = prompt;
                vid_gen_params.negative_prompt = negative_prompt;
                vid_gen_params.clip_skip = clip_skip;
                vid_gen_params.init_image = sample_init_image;
                vid_gen_params.end_image = end_image;
                vid_gen_params.control_frames = control_frames.data();
                vid_gen_params.control_frames_size = (int)control_frames.size();
                vid_gen_params.width = format_w;
                vid_gen_params.height = format_y;
                vid_gen_params.sample_params = sample_params;
                vid_gen_params.high_noise_sample_params = high_noise_sample_params;
                vid_gen_params.moe_boundary = moe_boundary;
                vid_gen_params.strength = strength;
                vid_gen_params.seed = seed;
                vid_gen_params.vace_strength = vace_strength;                
                vid_gen_params.video_frames = num_results;

                results     = generate_video(sd_ctx, &vid_gen_params, &num_results);
            }

            if(isConnected(0)){
                if(use_init_image_as_ref == false){
                    printf("Cleaning Input IMAGE\n");
                    free(init_image.data);
                }
                printf("Cleaning Mask\n");
                free(mask_image.data); 
            }
            if(isConnected(2)){
                printf("Cleaning Control IMAGE\n");
                free(control_image.data);
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
                    for(int i=0;i<num_results;i++){
                        save_image(results[i], temp_path, false);
                        printf("Saved temporal files to %s\n", temp_path);
                    }
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
        int result_index = (output_index < 0) ? 0 : ((output_index >= num_results-1) ? num_results-1 : output_index);
        const Format f = format();            // current output format
        const int fx = f.x(), fy = f.y(), fr = f.r(), ft = f.t();
        const int W  = fr - fx;
        const int H  = ft - fy;

        // clip to our bounds (defensive: engine can be called with larger x..r)
        int xi0 = (std::max)(x, fx);
        int xi1 = (std::min)(r, fr);
        if (xi0 >= xi1 || y < fy || y >= ft) return;

        const bool flipY = true;
        const int srcY   = flipY ? (H - 1 - (y - fy)) : (y - fy);

        size_t idx = (size_t(srcY) * W + (xi0 - fx)) * 3;

        // Grab writable channel pointers (only if requested)
        float* R = (channels & Mask_Red)   ? out.writable(Chan_Red)   + xi0 : nullptr;
        float* G = (channels & Mask_Green) ? out.writable(Chan_Green) + xi0 : nullptr;
        float* B = (channels & Mask_Blue)  ? out.writable(Chan_Blue)  + xi0 : nullptr;

        for (int xi = xi0; xi < xi1; ++xi) {
            float r8 = static_cast<float>(results[result_index].data[idx++]);
            float g8 = static_cast<float>(results[result_index].data[idx++]);
            float b8 = static_cast<float>(results[result_index].data[idx++]);

            if (R) *R++ = r8 / 255.f;
            if (G) *G++ = g8 / 255.f;
            if (B) *B++ = b8 / 255.f;
        }
    }

    void knobs(Knob_Callback f)
    {
        Enumeration_knob(f, &mode, modes_str, "mode", "mode");
        Enumeration_knob(f, &video_mode, video_modes_str, "video_mode", "video_mode");
        //const char* renderScript = "nuke.execute(nuke.thisNode(), nuke.frame(), nuke.frame(), 1)";
        const char* renderScript =  "n = nuke.thisNode()\n"
                                    "mode = n['mode'].value()\n"
                                    "video_mode = n['video_mode'].value()\n"
                                    "if mode == 'vid_gen' and video_mode == 'video2video':\n"
                                    "    first = int(n['frame_range'].value(0))\n"
                                    "    last  = int(n['frame_range'].value(1))\n"
                                    "else:\n"
                                    "    first = last = nuke.frame()\n"
                                    "nuke.execute(n, first, last, 1)\n";
        PyScript_knob(f, renderScript, "Execute");
        __formatknob = WH_knob(f,&format_w,"format","format");
        //SetFlags(f, Knob::SLIDER | Knob::NO_PROXYSCALE);
        ClearFlags(f, Knob::SLIDER | Knob::MAGNITUDE);
        
        MultiInt_knob(f, frame_range, 2, "frame_range", "frame_range");

        Int_knob(f, &output_index, "output_index", "output_index");
        SetFlags(f, Knob::SLIDER);

        BeginGroup(f, "Temp Files");
        Bool_knob(f, &temp_save, "save_temporal_files", "save_temporal_files");
        SetFlags(f, Knob::STARTLINE );  
        File_knob(f, &temp_path, "temp_path", "temp_path");
        SetFlags(f, Knob::STARTLINE );
        EndGroup(f);

        BeginGroup(f, "Prompt");
        Multiline_String_knob(f, &prompt, "prompt", "prompt",5 );
        Multiline_String_knob(f, &negative_prompt, "negative_prompt", "negative_prompt", 5 );
        EndGroup(f);

        BeginGroup(f, "Sample Params");
        Bool_knob(f, &use_init_image_as_ref, "use_init_image_as_ref", "use_init_image_as_ref");
        SetFlags(f, Knob::STARTLINE );

        Enumeration_knob(f, &sample_method, sample_method_names.data(), "sample_method", "sample_method");   
        Enumeration_knob(f, &schedule, schedule_names.data(), "schedule", "schedule");  

        Int_knob(f, &sample_steps, "sample_steps", "sample_steps");
        SetFlags(f, Knob::SLIDER);
    
        Int_knob(f, &seed, "seed", "seed");
        SetFlags(f, Knob::SLIDER);

        Float_knob(f, &txt_cfg, "txt_cfg", "txt_cfg");
        Float_knob(f, &img_cfg, "img_cfg", "img_cfg");
        Float_knob(f, &guidance, "guidance", "guidance");
        Float_knob(f, &strength, "strength", "strength");
        SetRange(f, 0.0, 1.0);

        Int_knob(f, &clip_skip, "clip_skip", "clip_skip");
        EndGroup(f);

        BeginGroup(f, "High Noise Params");
        Int_knob(f, &high_noise_sample_steps, "high_noise_sample_steps", "high_noise_sample_steps");
        SetFlags(f, Knob::SLIDER);            
        Float_knob(f, &high_noise_txt_cfg, "high_noise_txt_cfg", "high_noise_txt_cfg");
        Float_knob(f, &high_noise_img_cfg, "high_noise_img_cfg", "high_noise_img_cfg");
        Float_knob(f, &high_noise_guidance, "high_noise_guidance", "high_noise_guidance");
        EndGroup(f);

        BeginGroup(f, "skip layer guidance (SLG)");
        Float_knob(f, &slg_scale, "slg_scale", "slg_scale");
        Float_knob(f, &slg_layer_start, "slg_layer_start", "slg_layer_start");
        Float_knob(f, &slg_layer_end, "slg_layer_end", "slg_layer_end");

        Float_knob(f, &high_noise_slg_scale, "high_noise_slg_scale", "high_noise_slg_scale");
        Float_knob(f, &high_noise_slg_layer_start, "high_noise_slg_layer_start", "high_noise_slg_layer_start");
        Float_knob(f, &high_noise_slg_layer_end, "high_noise_slg_layer_end", "high_noise_slg_layer_end");
        EndGroup(f);


        BeginGroup(f, "Wan Video");
        Float_knob(f, &vace_strength, "vace_strength", "vace_strength");
        SetFlags(f, Knob::SLIDER);
        Float_knob(f, &moe_boundary, "moe_boundary", "moe_boundary");
        SetFlags(f, Knob::SLIDER);

        EndGroup(f);
  
        //Float_knob(f, &pm_style_strength, "photomaker_style_strength", "photomaker_style_strength");
        BeginGroup(f, "VAE Tiling");
        Bool_knob(f, &vae_tiling, "vae_tiling", "vae_tiling");
        SetFlags(f, Knob::STARTLINE );
        Int_knob(f, &tile_size, "vae_tile_size", "vae_tile_size");
        SetFlags(f, Knob::SLIDER);
        Float_knob(f, &tile_overlap, "vae_tile_overlap", "vae_tile_overlap");
        SetFlags(f, Knob::SLIDER);
        SetRange(f, 0.0, 1.0);
        EndGroup(f);

        BeginGroup(f, "ControlNet");    
        Float_knob(f, &control_strength, "control_strength", "control_strength");
        EndGroup(f);

        Int_knob(f, &token_, "token", "");               // Force update when changed
        SetFlags(f, Knob::INVISIBLE | Knob::NO_ANIMATION);        
    }    
    const char* Class() const { return desc.name; };
    const char* node_help() const { return HELP; };

    static const Iop::Description desc;  
};

inline void sd_prog_call(int step, int steps, float time, void* data){
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
    if( mode == VID_GEN && video_mode == V2V ){
        curr_frame = outputContext().frame();
        if (curr_frame < frame_range[0] || curr_frame > frame_range[1])
            return;
        if(curr_frame == frame_range[0]){
            // Clean up control frames
            printf("Cleaning Control Frames\n");
            for (auto& frame : control_frames) {
                if (frame.data) {
                    free(frame.data);
                    frame.data = NULL;
                }
            }
            control_frames.clear();            
        }  
                        
        // Control Video
        if(isConnected(2)){
            printf("Reading Control Frame %zu\n",curr_frame); 
            sd_images_out init_data = input2sdimages(input(2),(int)format_w,(int)format_y,false);
            control_frames.push_back(init_data.rgb);
        };
        if(curr_frame == frame_range[1]){
            sample_sd();
        }
    }else{
        sample_sd();
    }

    ++token_;
    Knob* t = knob("token");
    if (t) t->set_value(token_);        
}

static Iop* pc_sd_inferenceCreate(Node* node)
{
  return new pc_sd_inference(node);
}
const Iop::Description pc_sd_inference::desc(CLASS, "pc_nuke_diffusion", pc_sd_inferenceCreate);
