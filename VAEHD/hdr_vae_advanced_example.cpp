/**
 * Advanced HDR VAE Decode Example - Complete Implementation
 * 
 * This example demonstrates the complete HDR VAE decode functionality
 * with advanced feature extraction and intelligent HDR processing.
 * 
 * Key Features Demonstrated:
 * 1. Advanced pre-conv feature extraction for intelligent HDR processing
 * 2. Feature-guided HDR expansion using pseudo pre-conv features
 * 3. Professional HDR workflow with content analysis
 * 4. HDR quality validation and statistics
 * 5. Flux model detection and optimizations
 * 6. Multiple HDR modes with adaptive processing
 */

#include "hdr_vae_decode.hpp"
#include "stable_diffusion.h"
#include <iostream>
#include <memory>

/**
 * Example 1: Complete Professional HDR Workflow
 * This demonstrates the full professional HDR pipeline with all features
 */
void example_professional_hdr_workflow(std::shared_ptr<VAE> vae, struct ggml_tensor* latent) {
    std::cout << "\n=== Professional HDR Workflow Example ===\n";
    
    // Create work context
    struct ggml_init_params params;
    params.mem_size = 256 * 1024 * 1024; // 256MB
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    struct ggml_context* work_ctx = ggml_init(params);
    
    try {
        // Step 1: Analyze input latent characteristics
        std::cout << "Step 1: Analyzing input latent...\n";
        HDRStats input_stats = analyze_hdr_tensor(latent);
        std::cout << "  Input range: [" << input_stats.min_value << ", " << input_stats.max_value << "]\n";
        std::cout << "  Dynamic range: " << input_stats.dynamic_range << "\n";
        
        // Step 2: Detect model type for optimal processing
        bool is_flux = is_flux_model(vae);
        std::cout << "Step 2: Model detection - " << (is_flux ? "Flux" : "Standard") << " model detected\n";
        
        // Step 3: Recommend optimal HDR mode
        HDRMode recommended_mode = recommend_hdr_mode(input_stats, is_flux);
        std::cout << "Step 3: Recommended HDR mode - " << hdr_mode_to_string(recommended_mode) << "\n";
        
        // Step 4: Create advanced HDR decoder with recommended settings
        auto advanced_decoder = create_advanced_hdr_vae_decoder(
            vae, recommended_mode,
            50.0f,  // max_range
            1.0f,   // scale_factor
            false,  // enable_negatives
            true    // debug_mode
        );
        
        // Step 5: Perform advanced HDR decode with feature extraction
        std::cout << "Step 5: Performing advanced HDR decode...\n";
        struct ggml_tensor* hdr_result = advanced_decoder->decode_hdr_advanced(work_ctx, latent);
        
        if (hdr_result) {
            // Step 6: Analyze results
            HDRStats output_stats = analyze_hdr_tensor(hdr_result);
            std::cout << "Step 6: HDR decode results:\n";
            std::cout << "  Output range: [" << output_stats.min_value << ", " << output_stats.max_value << "]\n";
            std::cout << "  HDR pixels: " << output_stats.hdr_pixel_count << " (" << output_stats.hdr_percentage << "%)\n";
            std::cout << "  Super HDR pixels: " << output_stats.super_hdr_count << "\n";
            std::cout << "  Dynamic range: " << output_stats.dynamic_range << "\n";
            
            // Step 7: Validate HDR quality
            bool quality_ok = validate_hdr_quality(input_stats, output_stats);
            std::cout << "Step 7: HDR quality validation - " << (quality_ok ? "PASSED" : "FAILED") << "\n";
            
            // Step 8: Check if pre-conv features were captured
            struct ggml_tensor* features = advanced_decoder->get_last_pre_conv_features();
            if (features) {
                std::cout << "Step 8: Pre-conv features captured successfully\n";
                std::cout << "  Feature dimensions: " << features->ne[0] << "x" << features->ne[1] << "x" << features->ne[2] << "\n";
            } else {
                std::cout << "Step 8: Using pseudo feature generation\n";
            }
            
            std::cout << "Professional HDR workflow completed successfully!\n";
        } else {
            std::cout << "HDR decode failed!\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in professional workflow: " << e.what() << "\n";
    }
    
    ggml_free(work_ctx);
}

/**
 * Example 2: Feature-Guided HDR Processing Comparison
 * This demonstrates the difference between standard and feature-guided HDR
 */
void example_feature_guided_comparison(std::shared_ptr<VAE> vae, struct ggml_tensor* latent) {
    std::cout << "\n=== Feature-Guided HDR Comparison ===\n";
    
    struct ggml_init_params params;
    params.mem_size = 512 * 1024 * 1024; // 512MB for two decodes
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    struct ggml_context* work_ctx = ggml_init(params);
    
    try {
        // Standard HDR decode
        std::cout << "Performing standard HDR decode...\n";
        struct ggml_tensor* standard_result = hdr_vae_decode_extended(
            work_ctx, vae, latent, HDR_MODERATE, 50.0f, 1.0f, false, true);
        
        // Advanced feature-guided HDR decode
        std::cout << "Performing feature-guided HDR decode...\n";
        struct ggml_tensor* advanced_result = hdr_vae_decode_professional(
            work_ctx, vae, latent, HDR_MODERATE, 50.0f, 1.0f, false, true, false, true);
        
        if (standard_result && advanced_result) {
            HDRStats standard_stats = analyze_hdr_tensor(standard_result);
            HDRStats advanced_stats = analyze_hdr_tensor(advanced_result);
            
            std::cout << "Comparison Results:\n";
            std::cout << "  Standard HDR - Range: [" << standard_stats.min_value << ", " << standard_stats.max_value << "], HDR%: " << standard_stats.hdr_percentage << "\n";
            std::cout << "  Advanced HDR - Range: [" << advanced_stats.min_value << ", " << advanced_stats.max_value << "], HDR%: " << advanced_stats.hdr_percentage << "\n";
            
            float improvement = advanced_stats.dynamic_range / standard_stats.dynamic_range;
            std::cout << "  Dynamic range improvement: " << improvement << "x\n";
            
            if (improvement > 1.1f) {
                std::cout << "  Feature-guided processing provided significant improvement!\n";
            } else {
                std::cout << "  Results are similar - content may not benefit from advanced processing\n";
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error in comparison: " << e.what() << "\n";
    }
    
    ggml_free(work_ctx);
}

/**
 * Example 3: HDR Mode Optimization
 * This demonstrates automatic HDR mode selection based on content analysis
 */
void example_hdr_mode_optimization(std::shared_ptr<VAE> vae, struct ggml_tensor* latent) {
    std::cout << "\n=== HDR Mode Optimization ===\n";
    
    struct ggml_init_params params;
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    struct ggml_context* work_ctx = ggml_init(params);
    
    try {
        // Test all HDR modes and find the best one
        HDRMode modes[] = {HDR_CONSERVATIVE, HDR_MODERATE, HDR_EXPOSURE, HDR_AGGRESSIVE};
        const char* mode_names[] = {"Conservative", "Moderate", "Exposure", "Aggressive"};
        
        HDRStats best_stats = {};
        HDRMode best_mode = HDR_MODERATE;
        float best_score = 0.0f;
        
        for (int i = 0; i < 4; i++) {
            std::cout << "Testing " << mode_names[i] << " mode...\n";
            
            struct ggml_tensor* result = hdr_vae_decode_professional(
                work_ctx, vae, latent, modes[i], 50.0f, 1.0f, false, false);
            
            if (result) {
                HDRStats stats = analyze_hdr_tensor(result);
                
                // Calculate quality score (balance between HDR content and reasonable range)
                float score = stats.hdr_percentage * 0.4f + 
                             std::min(10.0f, stats.dynamic_range) * 0.3f +
                             (stats.has_significant_hdr ? 20.0f : 0.0f) * 0.3f;
                
                std::cout << "  Score: " << score << ", HDR%: " << stats.hdr_percentage << ", Range: " << stats.dynamic_range << "\n";
                
                if (score > best_score) {
                    best_score = score;
                    best_mode = modes[i];
                    best_stats = stats;
                }
            }
        }
        
        std::cout << "Optimal HDR mode: " << hdr_mode_to_string(best_mode) << " (score: " << best_score << ")\n";
        std::cout << "  Best results - HDR%: " << best_stats.hdr_percentage << ", Dynamic range: " << best_stats.dynamic_range << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error in optimization: " << e.what() << "\n";
    }
    
    ggml_free(work_ctx);
}

/**
 * Example 4: Batch HDR Processing with Quality Control
 */
void example_batch_hdr_processing(std::shared_ptr<VAE> vae, std::vector<struct ggml_tensor*> latents) {
    std::cout << "\n=== Batch HDR Processing ===\n";
    
    struct ggml_init_params params;
    params.mem_size = 1024 * 1024 * 1024; // 1GB for batch processing
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    struct ggml_context* work_ctx = ggml_init(params);
    
    try {
        std::cout << "Processing " << latents.size() << " latents in batch...\n";
        
        // Analyze all inputs first to determine optimal processing strategy
        std::vector<HDRStats> input_stats;
        HDRMode optimal_mode = HDR_MODERATE;
        
        for (size_t i = 0; i < latents.size(); i++) {
            HDRStats stats = analyze_hdr_tensor(latents[i]);
            input_stats.push_back(stats);
            
            // Update optimal mode based on content analysis
            HDRMode recommended = recommend_hdr_mode(stats, is_flux_model(vae));
            if (i == 0 || recommended == HDR_AGGRESSIVE) {
                optimal_mode = recommended; // Use most aggressive needed
            }
        }
        
        std::cout << "Optimal batch mode: " << hdr_mode_to_string(optimal_mode) << "\n";
        
        // Process batch with optimal settings
        std::vector<struct ggml_tensor*> results = hdr_vae_decode_batch(
            work_ctx, vae, latents, optimal_mode, 50.0f, 1.0f, false, true);
        
        // Quality control - analyze results
        int successful_count = 0;
        float total_improvement = 0.0f;
        
        for (size_t i = 0; i < results.size() && i < input_stats.size(); i++) {
            if (results[i]) {
                HDRStats output_stats = analyze_hdr_tensor(results[i]);
                bool quality_ok = validate_hdr_quality(input_stats[i], output_stats);
                
                if (quality_ok) {
                    successful_count++;
                    total_improvement += output_stats.dynamic_range / input_stats[i].dynamic_range;
                }
                
                std::cout << "  Item " << i << ": " << (quality_ok ? "OK" : "FAILED") 
                         << ", HDR%: " << output_stats.hdr_percentage << "\n";
            }
        }
        
        float success_rate = (float)successful_count / results.size() * 100.0f;
        float avg_improvement = total_improvement / successful_count;
        
        std::cout << "Batch processing complete:\n";
        std::cout << "  Success rate: " << success_rate << "%\n";
        std::cout << "  Average improvement: " << avg_improvement << "x\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error in batch processing: " << e.what() << "\n";
    }
    
    ggml_free(work_ctx);
}

/**
 * Main example function - demonstrates all advanced HDR features
 */
int main_hdr_advanced_example() {
    std::cout << "Advanced HDR VAE Decode - Complete Implementation Demo\n";
    std::cout << "====================================================\n";
    
    // Note: In a real application, you would load your VAE model and latents here
    // std::shared_ptr<VAE> vae = load_vae_model("path/to/vae");
    // struct ggml_tensor* latent = load_latent("path/to/latent");
    
    std::cout << "This example demonstrates the complete HDR VAE decode implementation\n";
    std::cout << "with advanced features including:\n\n";
    
    std::cout << "✓ Pre-conv feature extraction for intelligent HDR processing\n";
    std::cout << "✓ Feature-guided HDR expansion with pseudo pre-conv features\n";
    std::cout << "✓ Professional HDR workflow with content analysis\n";
    std::cout << "✓ HDR quality validation and statistics\n";
    std::cout << "✓ Flux model detection and optimizations\n";
    std::cout << "✓ Multiple HDR modes with adaptive processing\n";
    std::cout << "✓ Batch processing with quality control\n";
    std::cout << "✓ Comprehensive HDR analysis and optimization\n\n";
    
    std::cout << "Key Implementation Features:\n";
    std::cout << "- AdvancedHDRVAEDecoder with pre-conv feature capture\n";
    std::cout << "- Pseudo pre-conv feature generation for existing VAE models\n";
    std::cout << "- Feature-guided HDR expansion algorithms\n";
    std::cout << "- Smart pixel analysis with neighborhood context\n";
    std::cout << "- Professional workflow functions\n";
    std::cout << "- HDR statistics and quality validation\n";
    std::cout << "- Automatic mode recommendation system\n\n";
    
    std::cout << "To use this implementation in your project:\n";
    std::cout << "1. Include hdr_vae_decode.hpp in your project\n";
    std::cout << "2. Use hdr_vae_decode_professional() for best results\n";
    std::cout << "3. Use AdvancedHDRVAEDecoder for maximum control\n";
    std::cout << "4. Analyze results with HDRStats functions\n\n";
    
    std::cout << "Example usage:\n";
    std::cout << "  auto result = hdr_vae_decode_professional(ctx, vae, latent, HDR_MODERATE);\n";
    std::cout << "  HDRStats stats = analyze_hdr_tensor(result);\n";
    std::cout << "  bool quality_ok = validate_hdr_quality(input_stats, stats);\n\n";
    
    // Uncommment these calls when you have actual VAE and latent data:
    // example_professional_hdr_workflow(vae, latent);
    // example_feature_guided_comparison(vae, latent);
    // example_hdr_mode_optimization(vae, latent);
    // example_batch_hdr_processing(vae, {latent});
    
    return 0;
}