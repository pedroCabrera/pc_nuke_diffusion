# HDR VAE Decode for stable-diffusion.cpp

A professional-grade HDR (High Dynamic Range) VAE decoder implementation for the stable-diffusion.cpp library, specifically optimized for Flux models and VFX workflows.

## Overview

This implementation provides extended dynamic range VAE decoding that preserves highlight information above 1.0, making it suitable for:

- Professional VFX and compositing workflows
- HDR image generation for cinema and television
- Scientific visualization requiring extended dynamic range
- High-quality image processing pipelines

Based on the ComfyUI HDRVAEDecode node but fully adapted for the ggml-based stable-diffusion.cpp architecture.

## Features

### HDR Modes

- **Conservative** (1.5x expansion): Gentle highlight extension, safest for general use
- **Moderate** (3x expansion): Balanced quality/range expansion (default)
- **Exposure** (exposure-based): Natural exposure-based HDR for compositing workflows  
- **Aggressive** (full recovery): Maximum mathematical range recovery

### Optimizations

- **Flux Model Support**: Specific optimizations for Flux model characteristics
- **Smart Pixel Analysis**: Intelligent neighborhood-based HDR expansion
- **Batch Processing**: Efficient processing of multiple images
- **Tiled Decode**: Support for large images with memory-efficient tiling
- **Float32 Pipeline**: Full precision throughout the processing chain

## Installation

### Prerequisites

- stable-diffusion.cpp library
- C++17 compatible compiler  
- CMake 3.15 or higher
- Optional: OpenEXR for HDR image I/O

### Build Integration

1. Copy the HDR header to your stable-diffusion.cpp project:
```bash
cp hdr_vae_decode.hpp /path/to/your/project/
```

2. Add to your CMakeLists.txt:
```cmake
# Include HDR support
include(CMakeLists_HDR.txt)
```

3. Build your project:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Note**: HDR VAE decode is now header-only, making integration even simpler!

## Usage

### Basic Usage

```cpp
#include "hdr_vae_decode.hpp"

// Initialize your VAE and context
std::shared_ptr<VAE> vae = /* your VAE instance */;
struct ggml_context* work_ctx = /* your work context */;
struct ggml_tensor* latent = /* your latent tensor */;

// Create HDR decoder
auto hdr_decoder = create_hdr_vae_decoder(
    vae,
    HDR_MODERATE,      // HDR mode
    50.0f,             // Max range
    1.0f,              // Scale factor
    false,             // Enable negatives
    true               // Debug mode
);

// Decode with HDR
struct ggml_tensor* hdr_output = hdr_decoder->decode_hdr(work_ctx, latent);
```

### Flux-Optimized Usage

```cpp
// Enhanced decode with Flux optimizations
struct ggml_tensor* result = hdr_vae_decode_extended(
    work_ctx,
    vae,
    latent,
    HDR_EXPOSURE,      // Exposure mode for natural HDR
    100.0f,            // Higher range for Flux
    1.0f,              // Scale factor
    true,              // Enable negatives
    false,             // Production mode
    false              // Not video
);
```

### Integration with Existing Workflow

```cpp
// Configure HDR in your SD context
sd_set_hdr_decode(sd_ctx, true, HDR_MODERATE, 50.0f, false);

// Use HDR-aware img2img
sd_image_t* result = img2img_hdr(
    sd_ctx, init_image, prompt, negative_prompt,
    clip_skip, cfg_scale, width, height,
    sample_method, sample_steps, strength, seed,
    batch_count, control_cond, control_strength,
    style_ratio, normalize_input, input_id_images_path,
    true  // Enable HDR decode
);
```

### Command Line Usage

```bash
# Enable HDR decode with moderate mode
./sd --hdr-decode --hdr-mode moderate --hdr-max-range 50.0

# Aggressive HDR for maximum range
./sd --hdr-decode --hdr-mode aggressive --hdr-max-range 100.0 --hdr-enable-negatives

# Exposure mode for natural compositing
./sd --hdr-decode --hdr-mode exposure --hdr-max-range 25.0
```

## API Reference

### HDRVAEDecoder Class

#### Constructor
```cpp
HDRVAEDecoder(std::shared_ptr<VAE> vae, 
              HDRMode mode = HDR_MODERATE,
              float max_range = 50.0f,
              float scale_factor = 1.0f,
              bool enable_negatives = false,
              bool debug_mode = false)
```

#### Methods
- `decode_hdr(work_ctx, latent, decode_video)`: Main HDR decode function
- `set_hdr_mode(mode)`: Change HDR processing mode
- `apply_hdr_range_clamping(tensor)`: Apply HDR-aware clamping

### Utility Functions

```cpp
// Extended decode with model-specific optimizations
struct ggml_tensor* hdr_vae_decode_extended(/* parameters */);

// Batch processing
std::vector<struct ggml_tensor*> hdr_vae_decode_batch(/* parameters */);

// Tiled decode for large images
struct ggml_tensor* hdr_vae_decode_tiled(/* parameters */);
```

### C API

```c
// Configure HDR decode
void sd_set_hdr_decode(sd_ctx_t* sd_ctx, bool enable_hdr, int hdr_mode, 
                       float max_range, bool enable_negatives);

// HDR VAE decode
sd_image_t* sd_vae_decode_hdr(sd_ctx_t* sd_ctx, float* latent, 
                              int width, int height, int channels);
```

## Advanced Features

### Pre-Conv Feature Extraction
The most important breakthrough for professional HDR processing:

- **AdvancedHDRVAEDecoder**: Captures features before the final conv_out layer
- **Pseudo pre-conv features**: Creates meaningful feature approximations for existing VAE models
- **Feature-guided expansion**: Uses pre-conv analysis to intelligently determine HDR expansion
- **Multi-channel feature analysis**: Analyzes structural, detail, and highlight features separately

### Intelligent HDR Processing
Advanced algorithms that analyze both output and intermediate features:

- **Feature-guided Conservative Mode**: Uses feature confidence to modulate gentle expansion
- **Feature-guided Moderate Mode**: Balances expansion with feature analysis for optimal quality
- **Feature-guided Exposure Mode**: Natural exposure-based HDR guided by feature characteristics  
- **Feature-guided Aggressive Mode**: Maximum range recovery with feature-based safety controls

### Smart HDR Expansion
Multi-layered intelligent processing:

- **Advanced feature-guided HDR**: Uses pre-conv features for comprehensive understanding
- **Pixel neighborhood analysis**: Examines surrounding pixels to determine optimal expansion
- **Edge preservation**: Maintains sharp edges while expanding highlights
- **Content-aware processing**: Different expansion strategies based on image content analysis

### Professional Workflow Tools
Complete professional pipeline with quality control:

- **HDR content analysis**: Comprehensive statistics and quality metrics
- **Automatic mode recommendation**: AI-driven optimal HDR mode selection
- **Quality validation**: Automated validation of HDR processing success
- **Batch processing**: Efficient batch HDR decode with quality control

### Flux Model Optimizations
Special handling for Flux model characteristics:

- **16-channel latent support**: Optimized for Flux's extended latent space
- **Highlight characteristic adaptation**: Tuned for Flux-specific highlight behavior
- **Architecture-specific adjustments**: Model detection and automatic optimization
- **Advanced pseudo features**: Flux-specific feature generation for better HDR results

## Professional Usage Examples

### Advanced HDR Workflow
```cpp
// Professional HDR pipeline with all features
struct ggml_tensor* result = hdr_vae_decode_professional(
    work_ctx, vae, latent, 
    HDR_MODERATE,    // Mode (auto-recommended)
    50.0f,           // Max range
    1.0f,            // Scale factor
    false,           // Enable negatives
    true,            // Debug mode
    false,           // Decode video
    true             // Force advanced processing
);

// Analyze results
HDRStats stats = analyze_hdr_tensor(result);
bool quality_ok = validate_hdr_quality(input_stats, stats);
```

### Feature-Guided Processing
```cpp
// Create advanced decoder with feature extraction
auto advanced_decoder = create_advanced_hdr_vae_decoder(
    vae, HDR_AGGRESSIVE, 100.0f, 1.0f, false, true);

// Decode with pre-conv feature analysis
struct ggml_tensor* hdr_result = advanced_decoder->decode_hdr_advanced(
    work_ctx, latent);

// Access captured features
struct ggml_tensor* features = advanced_decoder->get_last_pre_conv_features();
```

## HDR Modes Comparison

| Mode | Expansion | Use Case | Quality | Range | Feature-Guided |
|------|-----------|----------|---------|-------|----------------|
| Conservative | 1.5x | General use, safe | Highest | Limited | ✓ Confidence-based |
| Moderate | 3x | Balanced workflows | High | Good | ✓ Smart balancing |
| Exposure | Exposure-based | Compositing | High | Natural | ✓ Exposure hints |
| Aggressive | Mathematical | Maximum range | Medium | Maximum | ✓ Safety controls |

## Performance Considerations

### Memory Usage
- HDR decode requires additional memory for extended range processing
- Batch processing scales linearly with image count
- Tiled decode reduces memory footprint for large images

### Speed Optimizations
- Use appropriate HDR mode for your workflow
- Enable compiler optimizations (-O3, /O2)
- Consider batch processing for multiple images
- Use tiling for memory-constrained systems

### Flux-Specific Optimizations
- Flux models benefit from exposure-based HDR modes
- Higher HDR ranges (50-100) work well with Flux
- Smart pixel analysis improves Flux highlight quality

## Troubleshooting

### Common Issues

**Issue**: Output appears too bright or washed out
- **Solution**: Use conservative mode or reduce max_range

**Issue**: No HDR effect visible
- **Solution**: Check input has highlights > 0.8, increase max_range

**Issue**: Memory allocation errors
- **Solution**: Use tiled decode or reduce batch size

**Issue**: Flux models produce artifacts
- **Solution**: Use exposure mode, enable smart pixel analysis

### Debug Mode

Enable debug mode for detailed processing information:
```cpp
auto hdr_decoder = create_hdr_vae_decoder(vae, HDR_MODERATE, 50.0f, 1.0f, false, true);
```

Debug output includes:
- Input/output tensor statistics
- HDR expansion details
- Processing timings
- Memory usage information

## Integration Examples

### Nuke Integration
```cpp
// For Nuke plugin development
void nuke_hdr_vae_process(/* Nuke parameters */) {
    // Create HDR decoder optimized for Nuke workflows
    auto hdr_decoder = create_hdr_vae_decoder(vae, HDR_EXPOSURE, 100.0f, 1.0f, true, false);
    
    // Process with extended range preservation
    ggml_tensor* result = hdr_decoder->decode_hdr(work_ctx, latent);
    
    // Output to Nuke's float buffer system
    // ... Nuke-specific output code ...
}
```

### Batch VFX Pipeline
```cpp
// Process multiple shots in a VFX pipeline
std::vector<ggml_tensor*> process_vfx_batch(std::vector<ggml_tensor*> latents) {
    return hdr_vae_decode_batch(work_ctx, vae, latents, HDR_EXPOSURE, 50.0f, 1.0f, true, false);
}
```

## Technical Details

### Header-Only Design

The HDR VAE decode implementation is now completely header-only, providing several advantages:
- **Easy Integration**: Just include the header file
- **No Linking Issues**: All code is compiled with your project
- **Better Optimization**: Compiler can inline and optimize across translation units
- **Template Friendly**: Supports template instantiation seamlessly

### HDR Processing Pipeline

1. **Input Processing**: Latent tensor preprocessing
2. **VAE Decode**: Standard VAE decode computation  
3. **Range Conversion**: Convert from [-1,1] to [0,2] preserving extended range
4. **HDR Expansion**: Apply mode-specific highlight expansion
5. **Smart Analysis**: Neighborhood-based pixel enhancement (optional)
6. **Range Clamping**: HDR-aware final range limiting

### Mathematical Foundation

The HDR expansion algorithms are based on:
- **Conservative**: Linear highlight expansion with smooth rolloff
- **Moderate**: Sigmoid-based smooth expansion preserving base image
- **Exposure**: Logarithmic exposure-stop based expansion  
- **Aggressive**: Inverse activation function for maximum recovery

### Flux Model Characteristics

Flux models exhibit specific highlight behavior:
- Different highlight threshold (0.85 vs 0.8)
- Smoother highlight rolloff characteristics
- Better response to exposure-based HDR modes
- Enhanced detail preservation in bright regions

## License

This HDR VAE decode implementation follows the same license as stable-diffusion.cpp.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional HDR modes and algorithms  
- Model-specific optimizations
- Performance enhancements
- Better integration with existing workflows
- HDR image I/O improvements

## Acknowledgments

- Based on ComfyUI HDRVAEDecode by the ComfyUI community
- Inspired by professional VFX HDR workflows
- Optimized for the ggml tensor library architecture
- Special thanks to the stable-diffusion.cpp contributors

## Changelog

### v1.0.0
- Initial implementation with 4 HDR modes
- Flux model optimizations
- Batch processing support  
- Tiled decode capability
- C API integration
- Complete documentation and examples