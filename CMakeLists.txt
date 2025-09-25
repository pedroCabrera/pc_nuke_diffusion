cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

add_subdirectory(packages/stable-diffusion.cpp)

project(pc_nuke_diffusion LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH "CMake;${CMAKE_MODULE_PATH}")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_definitions(-DSD_CUDA -DSD_USE_CUDA)


find_package(Nuke REQUIRED)

function(add_plugin PLUGIN_NAME)
    add_nuke_plugin(${PLUGIN_NAME} ${ARGN})
    target_compile_definitions(${PLUGIN_NAME} PRIVATE FN_EXAMPLE_PLUGIN)
    # Link the stable-diffusion static library
    target_link_libraries(${PLUGIN_NAME}  PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${PLUGIN_NAME}  PUBLIC c_std_11 cxx_std_17)    
endfunction()

add_plugin(pc_sd_load_model   pc_sd_load_model.cpp)
add_plugin(pc_sd_inference    pc_sd_inference.cpp)
add_plugin(pc_sd_upscaler     pc_sd_upscaler.cpp)

# ---------------------------------------------------------------------------------
# Your CLI example (unchanged)
set(TARGET sd)
add_executable(${TARGET} "packages/stable-diffusion.cpp/examples/cli/main.cpp")
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
# ---------------------------------------------------------------------------------

#add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)
set(MAXINE_SDK_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/packages/Maxine-VFX-SDK"
    CACHE PATH "Path to the NVIDIA Maxine VFX SDK (OSS) repository")

# Build the proxy as a static lib so includes propagate cleanly
add_library(nvvfx_proxy STATIC
    "${MAXINE_SDK_DIR}/nvvfx/src/nvVideoEffectsProxy.cpp"
    "${MAXINE_SDK_DIR}/nvvfx/src/nvCVImageProxy.cpp"
)
target_include_directories(nvvfx_proxy PUBLIC "${MAXINE_SDK_DIR}/nvvfx/include" )
target_compile_features(nvvfx_proxy PUBLIC c_std_11 cxx_std_17)

add_nuke_plugin(pc_sd_nvfx_videoeffects pc_sd_nvfx_videoeffects.cpp )
target_compile_definitions(pc_sd_nvfx_videoeffects PRIVATE FN_EXAMPLE_PLUGIN)
# Link the stable-diffusion static library
target_compile_features(pc_sd_nvfx_videoeffects  PUBLIC c_std_11 cxx_std_17)   
# Link the Maxine proxy + include dirs
target_link_libraries(pc_sd_nvfx_videoeffects PUBLIC nvvfx_proxy)

# Delay-load the runtime DLLs (so the plugin won’t block Nuke if Maxine isn’t installed)
if (WIN32)
    target_link_options(pc_sd_nvfx_videoeffects PRIVATE
        /DELAYLOAD:NvVideoEffects.dll
        /DELAYLOAD:NvCVImage.dll
    )
    # MSVC needs delayimp for /DELAYLOAD
    if (MSVC)
        target_link_libraries(pc_sd_nvfx_videoeffects PRIVATE delayimp)
    endif()
endif()



