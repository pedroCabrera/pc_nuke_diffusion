# pc_nuke_diffusion
The Foundry Nuke implementation of diffusion models in c++ thansk to  https://github.com/leejet/stable-diffusion.cpp

***Note that this project, and mostly [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) are under active development.***

[Currently Supported models:  
SD1.x, SD2.x, SD-Turbo  
SDXL, SDXL-Turbo  
!!!The VAE in SDXL encounters NaN issues under FP16, but unfortunately, the ggml_conv_2d only operates under FP16. Hence, a parameter is needed to specify the VAE that has fixed the FP16 NaN issue. You can find it here: SDXL VAE FP16 Fix.
SD3/SD3.5  
Flux-dev/Flux-schnell  
Chroma  
FLUX.1-Kontext-dev  
Wan2.1/Wan2.2](https://github.com/leejet/stable-diffusion.cpp#:~:text=Image%20Models,Wan2.1/Wan2.2)

I highly recommend checking the [stable-diffusion.cpp documentation](https://github.com/leejet/stable-diffusion.cpp/tree/master/docs)

Also Support for some [Nvidia Maxine](https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-system-guide/index.html) Filters:

- **Encoder Artifact Reduction (Beta)**, which reduces the blocking and noisy artifacts that are produced from encoding while preserving the details of the original video.
The ArtifactReduction effect has the following modes:
    Strength 0, which applies a weak effect.
    Strength 1, which applies a strong effect.
    
- **Super resolution (Beta)**, which upscales a video and reduces encoding artifacts. This filter enhances the details, sharpens the output, and preserves the content. The SuperRes effect has two modes:
    Strength 1, which applies strong enhancements.
    Strength 0, which applies weaker enhancements while reducing encoding artifacts.
    
- **Upscale (Beta)**, which is a fast and light-weight method to upscale for an input video and sharpen the resulting output.
This filter can optionally be pipelined with encoder artifact reduction to enhance the scale while reducing the video artifacts.

Will add more nuke specific documentation in the future

