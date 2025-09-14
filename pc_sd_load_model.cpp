#include "DDImage/NoIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"
#include <memory>

#include "DDImage/Executable.h"

// Include Stable Diffusion headers
#include "stable-diffusion.h"
#include "stable_diffusion_wrapper.h"


using namespace DD::Image;

static const char* const CLASS = "pc_sd_load_model";
static const char* const HELP = "Stable Diffusion For Nuke";



class pc_sd_load_model : public NoIop,  Executable, public AbstractDataInterface {

    const char* model_path = "[python {os.getenv('STD_MODEL') + '/v1-5-pruned-emaonly.safetensors'}]";
    const char* diffusion_model_path = "";
    const char* high_noise_diffusion_model_path = "";

    const char* clip_l_path = "";
    const char* clip_g_path = "";
    const char* clip_vision_path = "";
    const char* t5xxl_path = "";

    const char* vae_path = "";
    const char* taesd_path = "";
    const char* controlnet_path = "";

    const char* lora_model_dir = "[python {os.getenv('STD_LORAS')}]";

    bool convert_weights = false;
    int weight_type = 0;

    bool vae_decode_only = false;
    bool offload_params_to_cpu = false;
    bool keep_clip_on_cpu = false;
    bool keep_control_net_on_cpu = false;
    bool keep_vae_on_cpu = false;
    bool diffusion_flash_attn = false;

    bool diffusion_conv_direct = false;
    bool vae_conv_direct = false;

    bool chroma_use_dit_mask = true;
    bool chroma_use_t5_mask = false;
    int chroma_t5_mask_pad = 1;
    float flow_shift;

    sd_ctx_t* sd_ctx ;

    bool clear_ctxt;

    const char* internaldata;
    virtual Executable* executable() { return this; } 

public:
    pc_sd_load_model(Node* node) : NoIop(node), Executable(this) , sd_ctx(nullptr) {
        
    }

    bool debug_mode = true;
    std::atomic<unsigned> m_progress{0};
    std::string m_progress_message;

    bool useStripes() { return false; };
    virtual bool renderFullPlanes() const { return true; };    
    void renderStripe( ImagePlane& outputPlane) {}
    void beginExecuting(){}
    void endExecuting(){}
    void execute();

    unsigned int getFrameProgress() const override {
        return m_progress.load(std::memory_order_relaxed);
    }
    std::string getFrameProgressMessage() const override {
        return m_progress_message;
    }

    void create_context() {
        if (sd_ctx != NULL) {
            free_sd_ctx(sd_ctx);
            sd_ctx = NULL;
        }
        if (sd_ctx == NULL) {
            printf("creating new sd_context\n");
            
            sd_ctx_params_t sd_ctx_params;
            sd_ctx_params_init(&sd_ctx_params);
            
            sd_ctx_params.model_path = model_path;
            sd_ctx_params.clip_l_path = clip_l_path;
            sd_ctx_params.clip_g_path = clip_g_path;
            sd_ctx_params.clip_vision_path = clip_vision_path;
            sd_ctx_params.t5xxl_path = t5xxl_path;
            sd_ctx_params.diffusion_model_path = diffusion_model_path;
            sd_ctx_params.high_noise_diffusion_model_path = high_noise_diffusion_model_path;
            sd_ctx_params.vae_path = vae_path;
            sd_ctx_params.taesd_path = taesd_path;
            sd_ctx_params.control_net_path = controlnet_path;
            sd_ctx_params.lora_model_dir = lora_model_dir;
            sd_ctx_params.embedding_dir = "";//embedding_dir;
            sd_ctx_params.stacked_id_embed_dir = "";//stacked_id_embed_dir;
            sd_ctx_params.vae_decode_only = vae_decode_only;//vae_decode_only;
            //sd_ctx_params.vae_tiling = false;//vae_tiling;
            sd_ctx_params.free_params_immediately = false;//Free_params_inmediately;
            sd_ctx_params.n_threads = get_num_physical_cores();//n_threads;
            sd_ctx_params.wtype = SD_TYPE_COUNT;//wtype;
            sd_ctx_params.rng_type = CUDA_RNG;
            sd_ctx_params.offload_params_to_cpu = offload_params_to_cpu;
            sd_ctx_params.keep_clip_on_cpu = keep_clip_on_cpu;
            sd_ctx_params.keep_control_net_on_cpu = keep_control_net_on_cpu;
            sd_ctx_params.keep_vae_on_cpu = keep_vae_on_cpu;
            sd_ctx_params.diffusion_flash_attn = diffusion_flash_attn;
            sd_ctx_params.diffusion_conv_direct = diffusion_conv_direct;
            sd_ctx_params.vae_conv_direct = vae_conv_direct;
            sd_ctx_params.chroma_use_dit_mask = chroma_use_dit_mask;
            sd_ctx_params.chroma_use_t5_mask = chroma_use_t5_mask;
            sd_ctx_params.chroma_t5_mask_pad = chroma_t5_mask_pad;
            sd_ctx_params.flow_shift = INFINITY;//flow_shift;

            sd_ctx = new_sd_ctx(&sd_ctx_params);
            
            if (sd_ctx == NULL) {
                printf("new_sd_ctx_t failed\n");
            }  
        }
    }

    const int get_weight_type(int  type){
        //
        if (type == 0) {
            return SD_TYPE_F32;
        } else if (type == 1) {
            return SD_TYPE_F16;
        } else if (type == 2) {
            return SD_TYPE_Q4_0;
        } else if (type == 3) {
            return SD_TYPE_Q4_1;
        } else if (type == 4) {
            return SD_TYPE_Q5_0;
        } else if (type == 5) {
            return SD_TYPE_Q5_1;
        } else if (type == 6) {
            return SD_TYPE_Q8_0;
        };
    }

    void _validate(bool for_real)
    {
        Format *customFormat = new Format(int(512), int(512));
        info_.format(*customFormat);
        info_.full_size_format(*customFormat);
        info_.x(0);
        info_.y(0);
        info_.r(int(512));
        info_.t(int(512));
        info_.turn_on(Mask_RGBA);    
    }

    sd_ctx_t* get_ctxt() const override {
        return sd_ctx;
    }


    void knobs(Knob_Callback f) override {
    const char* renderScript = "nuke.execute(nuke.thisNode(), nuke.frame(), nuke.frame(), 1)";
    PyScript_knob(f, renderScript, "Load Model");
    Bool_knob(f, &debug_mode, "debug_mode", "debug_mode");    
    SetFlags(f, Knob::STARTLINE );     
    Bool_knob(f, &clear_ctxt,"clear","clear");

    BeginGroup(f, "Model Files");
    File_knob(f, &model_path, "model", "model");
    File_knob(f, &diffusion_model_path, "diffusion_model", "diffusion_model");
    File_knob(f, &high_noise_diffusion_model_path, "high_noise_diffusion_model", "high_noise_diffusion_model");

    File_knob(f, &clip_l_path, "clip_l", "clip_l");
    File_knob(f, &clip_g_path, "clip_g", "clip_g");
    File_knob(f, &clip_vision_path, "clip_vision", "clip_vision");
    File_knob(f, &t5xxl_path, "t5xxl", "t5xxl");

    File_knob(f, &vae_path, "vae", "vae");
    File_knob(f, &taesd_path, "taesd", "taesd");
    File_knob(f, &controlnet_path, "controlnet", "controlnet");

    File_knob(f, &lora_model_dir, "lora_models_directory", "lora_models_directory");
    EndGroup(f);

    Bool_knob(f, &vae_decode_only, "vae_decode_only", "vae_decode_only"); 
    SetFlags(f, Knob::STARTLINE );

    BeginGroup(f, "Optimization");
    Bool_knob(f, &convert_weights, "convert_weights", "convert_weights");
    SetFlags(f, Knob::STARTLINE ); 
    Enumeration_knob(f, &weight_type, weight_type_names, "weight_type", "weight_type"); 
    
    Bool_knob(f, &offload_params_to_cpu, "offload_params_to_cpu", "offload_params_to_cpu");
    SetFlags(f, Knob::STARTLINE );    
    Bool_knob(f, &keep_clip_on_cpu, "keep_clip_on_cpu", "keep_clip_on_cpu");
    SetFlags(f, Knob::STARTLINE ); 
    Bool_knob(f, &keep_control_net_on_cpu, "keep_control_net_on_cpu", "keep_control_net_on_cpu");
    SetFlags(f, Knob::STARTLINE ); 
    Bool_knob(f, &keep_vae_on_cpu, "keep_vae_on_cpu", "keep_vae_on_cpu");    
    SetFlags(f, Knob::STARTLINE );

    Bool_knob(f, &diffusion_flash_attn, "diffusion_flash_attn", "diffusion_flash_attn");    
    SetFlags(f, Knob::STARTLINE ); 

    Bool_knob(f, &diffusion_conv_direct, "diffusion_conv_direct", "diffusion_conv_direct");    
    SetFlags(f, Knob::STARTLINE ); 
    Bool_knob(f, &vae_conv_direct, "vae_conv_direct", "vae_conv_direct");    
    SetFlags(f, Knob::STARTLINE ); 
    EndGroup(f);
    BeginGroup(f, "Advanced Chroma Options");
    SetFlags(f, Knob::STARTLINE );
    Bool_knob(f, &chroma_use_dit_mask, "chroma_use_dit_mask", "chroma_use_dit_mask");    
    SetFlags(f, Knob::STARTLINE );
    Bool_knob(f, &chroma_use_t5_mask, "chroma_use_t5_mask", "chroma_use_t5_mask");    
    SetFlags(f, Knob::STARTLINE );
    Int_knob(f, &chroma_t5_mask_pad, "chroma_t5_mask_pad", "chroma_t5_mask_pad");    
    SetFlags(f, Knob::STARTLINE );
    EndGroup(f);

    //Float_knob(f, &flow_shift, "flow_shift", "flow_shift");    
    //SetFlags(f, Knob::STARTLINE );
    }
    
  const char* Class() const { return desc.name; };
  const char* node_help() const { return HELP; };

  static const Iop::Description desc;  
};
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    pc_sd_load_model* node = (pc_sd_load_model*)data;
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log || (!node->debug_mode && level <= SD_LOG_DEBUG)) {
        return;
    }
    switch (level) {
        case SD_LOG_DEBUG:
            tag_color = 37;
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            tag_color = 34;
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            tag_color = 35;
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            tag_color = 31;
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            tag_color = 33;
            level_str = "?????";
            break;
    }

    fprintf(out_stream, "[%-5s] ", level_str);

    fputs(log, out_stream);
    fflush(out_stream);
    node->m_progress_message =log;
}

void sd_prog_call(int step, int steps, float time, void* data){
    unsigned p = static_cast<unsigned>((step) * 100 / steps);
    pc_sd_load_model* node = (pc_sd_load_model*)data;
    node->m_progress.store(p, std::memory_order_relaxed);
    nuke_pretty_progress(step, steps, time);

}
void pc_sd_load_model::execute() {
    sd_set_log_callback(sd_log_cb, this);
    sd_set_progress_callback(sd_prog_call, this);
    m_progress.store(0, std::memory_order_relaxed);
    m_progress_message = "Loading Model";
    if (clear_ctxt) {
        if (sd_ctx != NULL) {
            free_sd_ctx(sd_ctx);
            sd_ctx = NULL;
        }
    }else{
        create_context();
    }
}

static Op* buildCustomNode(Node* node) { 
  return new pc_sd_load_model(node);
}
const Op::Description pc_sd_load_model::desc(CLASS, "pc_nuke_diffusion", buildCustomNode);