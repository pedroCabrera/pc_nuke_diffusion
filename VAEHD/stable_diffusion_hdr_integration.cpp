/**
 * HDR VAE Decode Integration Patch for stable-diffusion.cpp
 * 
 * This file shows the modifications needed to integrate HDR VAE decode
 * functionality into the existing stable-diffusion.cpp codebase.
 * 
 * Add these modifications to your stable-diffusion.cpp file to enable HDR support.
 */

// Add this include at the top of stable-diffusion.cpp
#include "hdr_vae_decode.hpp"

// Add these members to the StableDiffusionGGML class
class StableDiffusionGGML {
    // ... existing members ...
    
    // HDR VAE decode support
    std::shared_ptr<HDRVAEDecoder> hdr_decoder = nullptr;
    bool use_hdr_decode = false;
    HDRMode hdr_mode = HDR_MODERATE;
    float hdr_max_range = 50.0f;
    bool hdr_enable_negatives = false;
    
    // ... rest of existing class ...
    
public:
    // Add HDR configuration method
    void configure_hdr_decode(bool enable_hdr = true,
                             HDRMode mode = HDR_MODERATE,
                             float max_range = 50.0f,
                             bool enable_negatives = false) {
        use_hdr_decode = enable_hdr;
        hdr_mode = mode;
        hdr_max_range = max_range;
        hdr_enable_negatives = enable_negatives;
        
        if (enable_hdr && first_stage_model != nullptr) {
            // Create HDR decoder when VAE is available
            hdr_decoder = create_hdr_vae_decoder(
                first_stage_model, mode, max_range, 1.0f, enable_negatives, false
            );
            LOG_INFO("HDR VAE decode configured: mode=%s, max_range=%.1f", 
                     hdr_decoder->get_hdr_mode_name(), max_range);
        }
    }
    
    // Modified decode_first_stage method with HDR support
    ggml_tensor* decode_first_stage_hdr(ggml_context* work_ctx, ggml_tensor* x, bool decode_video = false) {
        // If HDR decode is enabled and available, use it
        if (use_hdr_decode && hdr_decoder != nullptr) {
            LOG_DEBUG("Using HDR VAE decode");
            return hdr_decoder->decode_hdr(work_ctx, x, decode_video);
        }
        
        // Otherwise, fall back to standard decode
        return decode_first_stage(work_ctx, x, decode_video);
    }
    
    // Enhanced decode with Flux optimizations
    ggml_tensor* decode_first_stage_flux_hdr(ggml_context* work_ctx, ggml_tensor* x, bool decode_video = false) {
        if (use_hdr_decode && version == VERSION_FLUX_DEV) {
            LOG_DEBUG("Using Flux-optimized HDR VAE decode");
            
            return hdr_vae_decode_extended(
                work_ctx,
                first_stage_model,
                x,
                hdr_mode,
                hdr_max_range,
                1.0f,                   // Scale factor
                hdr_enable_negatives,
                false,                  // Debug mode
                decode_video
            );
        }
        
        // Fall back to regular HDR or standard decode
        return decode_first_stage_hdr(work_ctx, x, decode_video);
    }
};

// Modified process_vae_output_tensor for HDR support
__STATIC_INLINE__ void process_hdr_vae_output_tensor(struct ggml_tensor* src, float max_range = 50.0f, bool enable_negatives = false) {
    int64_t nelements = ggml_nelements(src);
    float* data = (float*)src->data;
    
    // Convert from [-1, 1] to [0, 2] preserving extended range
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        data[i] = (val + 1.0f) * 0.5f;
    }
    
    // Apply HDR-aware clamping
    float min_clamp = enable_negatives ? -max_range : 0.0f;
    float max_clamp = max_range;
    
    for (int i = 0; i < nelements; i++) {
        float val = data[i];
        val = std::max(min_clamp, std::min(max_clamp, val));
        data[i] = val;
    }
}

// Add these functions to the C API (stable-diffusion.h modifications)

// Add to sd_ctx_params_t struct:
typedef struct {
    // ... existing fields ...
    
    // HDR VAE decode parameters
    bool hdr_decode_enabled;
    int hdr_mode;              // 0=conservative, 1=moderate, 2=exposure, 3=aggressive
    float hdr_max_range;
    bool hdr_enable_negatives;
} sd_ctx_params_t;

// Add HDR decode functions to the C API
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configure HDR VAE decode for the context
 */
STABLE_DIFFUSION_API void sd_set_hdr_decode(sd_ctx_t* sd_ctx,
                                           bool enable_hdr,
                                           int hdr_mode,
                                           float max_range,
                                           bool enable_negatives);

/**
 * Decode latent with HDR preservation
 */
STABLE_DIFFUSION_API sd_image_t* sd_vae_decode_hdr(sd_ctx_t* sd_ctx,
                                                  float* latent,
                                                  int width,
                                                  int height,
                                                  int channels);

#ifdef __cplusplus
}
#endif

// Implementation of C API functions
void sd_set_hdr_decode(sd_ctx_t* sd_ctx,
                      bool enable_hdr,
                      int hdr_mode,
                      float max_range,
                      bool enable_negatives) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return;
    }
    
    HDRMode mode = static_cast<HDRMode>(hdr_mode);
    sd_ctx->sd->configure_hdr_decode(enable_hdr, mode, max_range, enable_negatives);
}

sd_image_t* sd_vae_decode_hdr(sd_ctx_t* sd_ctx,
                             float* latent,
                             int width,
                             int height,
                             int channels) {
    if (sd_ctx == nullptr || sd_ctx->sd == nullptr) {
        return nullptr;
    }
    
    // Create work context
    struct ggml_init_params params;
    params.mem_size = static_cast<size_t>(width * height * channels * sizeof(float) * 10); // Extra memory for processing
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    
    struct ggml_context* work_ctx = ggml_init(params);
    if (work_ctx == nullptr) {
        return nullptr;
    }
    
    // Create latent tensor
    struct ggml_tensor* latent_tensor = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 
                                                          width, height, channels, 1);
    
    // Copy latent data
    memcpy(latent_tensor->data, latent, ggml_nbytes(latent_tensor));
    
    // Decode with HDR
    struct ggml_tensor* result = sd_ctx->sd->decode_first_stage_flux_hdr(work_ctx, latent_tensor);
    
    if (result == nullptr) {
        ggml_free(work_ctx);
        return nullptr;
    }
    
    // Convert to sd_image_t
    sd_image_t* image = (sd_image_t*)malloc(sizeof(sd_image_t));
    image->width = result->ne[0];
    image->height = result->ne[1];
    image->channel = result->ne[2];
    
    size_t image_size = image->width * image->height * image->channel * sizeof(uint8_t);
    image->data = (uint8_t*)malloc(image_size);
    
    // Convert float HDR data to uint8_t (with HDR->LDR tone mapping if needed)
    float* hdr_data = (float*)result->data;
    for (int i = 0; i < image->width * image->height * image->channel; i++) {
        float val = hdr_data[i];
        
        // Simple tone mapping for display (you might want more sophisticated tone mapping)
        if (val > 1.0f) {
            val = 1.0f - expf(-val); // Exponential tone mapping
        }
        
        image->data[i] = (uint8_t)(val * 255.0f);
    }
    
    ggml_free(work_ctx);
    return image;
}

// Example modification to img2img function to support HDR
sd_image_t* img2img_hdr(sd_ctx_t* sd_ctx,
                       sd_image_t init_image,
                       const char* prompt,
                       const char* negative_prompt,
                       int clip_skip,
                       float cfg_scale,
                       int width,
                       int height,
                       enum sample_method_t sample_method,
                       int sample_steps,
                       float strength,
                       int64_t seed,
                       int batch_count,
                       const sd_image_t* control_cond,
                       float control_strength,
                       float style_ratio,
                       bool normalize_input,
                       const char* input_id_images_path,
                       bool use_hdr_decode) {
    
    // Configure HDR decode if requested
    if (use_hdr_decode) {
        sd_set_hdr_decode(sd_ctx, true, HDR_MODERATE, 50.0f, false);
    }
    
    // Call existing img2img function
    return img2img(sd_ctx, init_image, prompt, negative_prompt, clip_skip, cfg_scale,
                   width, height, sample_method, sample_steps, strength, seed, batch_count,
                   control_cond, control_strength, style_ratio, normalize_input, input_id_images_path);
}

// Add HDR parameter parsing to CLI arguments
// This would go in your main.cpp or CLI parsing code

void add_hdr_cli_options() {
    // Add these command line options:
    // --hdr-decode: Enable HDR VAE decode
    // --hdr-mode: HDR mode (conservative, moderate, exposure, aggressive)
    // --hdr-max-range: Maximum HDR range (default: 50.0)
    // --hdr-enable-negatives: Enable negative values
    
    // Example parsing:
    /*
    if (arg == "--hdr-decode") {
        params.hdr_decode_enabled = true;
    } else if (arg == "--hdr-mode") {
        std::string mode = argv[++i];
        if (mode == "conservative") params.hdr_mode = 0;
        else if (mode == "moderate") params.hdr_mode = 1;
        else if (mode == "exposure") params.hdr_mode = 2;
        else if (mode == "aggressive") params.hdr_mode = 3;
    } else if (arg == "--hdr-max-range") {
        params.hdr_max_range = std::stof(argv[++i]);
    } else if (arg == "--hdr-enable-negatives") {
        params.hdr_enable_negatives = true;
    }
    */
}

// Example usage in main function
/*
int main(int argc, char* argv[]) {
    // ... existing initialization code ...
    
    // Parse HDR options
    add_hdr_cli_options();
    
    // Initialize context with HDR support
    sd_ctx_t* sd_ctx = new_sd_ctx(model_path, params, vae_decode_only, 
                                  vae_tiling, control_net_cpu, n_threads, 
                                  wtype, rng_type, schedule, keep_control_net_cpu);
    
    // Configure HDR if enabled
    if (params.hdr_decode_enabled) {
        sd_set_hdr_decode(sd_ctx, true, params.hdr_mode, 
                         params.hdr_max_range, params.hdr_enable_negatives);
        printf("HDR VAE decode enabled: mode=%d, max_range=%.1f\n", 
               params.hdr_mode, params.hdr_max_range);
    }
    
    // ... rest of main function ...
}
*/