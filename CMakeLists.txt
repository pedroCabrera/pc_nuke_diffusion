cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(CUDAToolkit REQUIRED)
set(_cuda_runtime CUDA::cudart_static)

# Point to the deps install dir (set from your build.bat per earlier steps)
set(DEPS_INSTALL_DIR "" CACHE PATH "Prebuilt deps install root")

add_library(pc_deps INTERFACE)
target_link_libraries(pc_deps INTERFACE
  ${_cuda_runtime}
  CUDA::cuda_driver
  CUDA::cublas
  # CUDA::cublasLt  # uncomment if you later get cublasLt* unresolveds
)
target_include_directories(pc_deps INTERFACE
  "${DEPS_INSTALL_DIR}/include"
)

# Pick up all needed .lib files (adjust patterns if you add more)
file(GLOB _deps_libs
  "${DEPS_INSTALL_DIR}/lib/*.lib"
)

# Link them all transitively via the interface
target_link_libraries(pc_deps INTERFACE ${_deps_libs})
target_link_libraries(pc_deps INTERFACE ${_cuda_runtime_target})

project(pc_nuke_diffusion LANGUAGES C CXX CUDA)
set(PLUGIN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nuke/pc_nuke_diffusion/plugin_libs/${NUKE_VERSION}")

set(CMAKE_MODULE_PATH "CMake;${CMAKE_MODULE_PATH}")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(Nuke REQUIRED)


function(add_plugin PLUGIN_NAME)
    add_nuke_plugin(${PLUGIN_NAME} ${ARGN})
    target_compile_definitions(${PLUGIN_NAME} PRIVATE FN_EXAMPLE_PLUGIN)
    # Link the stable-diffusion static library
    target_link_libraries(${PLUGIN_NAME}  PRIVATE   pc_deps  ${CMAKE_THREAD_LIBS_INIT})
    #target_link_libraries(${PLUGIN_NAME} PUBLIC ${nvvfx_proxy_target})
    target_compile_features(${PLUGIN_NAME}  PUBLIC c_std_11 cxx_std_17)  
    # Delay-load the runtime DLLs (so the plugin won’t block Nuke if Maxine isn’t installed)
    if (WIN32)
        target_link_options(${PLUGIN_NAME} PRIVATE
            /DELAYLOAD:NvVideoEffects.dll
            /DELAYLOAD:NvCVImage.dll
        )
        # MSVC needs delayimp for /DELAYLOAD
        if (MSVC)
            target_link_libraries(${PLUGIN_NAME} PRIVATE delayimp)
        endif()
    endif()    
    install(TARGETS ${PLUGIN_NAME}
    RUNTIME DESTINATION ${PLUGIN_OUTPUT_DIR}
    LIBRARY DESTINATION ${PLUGIN_OUTPUT_DIR}
    ARCHIVE DESTINATION ${PLUGIN_OUTPUT_DIR}    
    )
endfunction()

add_plugin(pc_nuke_diffusion   pc_sd_load_model.cpp pc_sd_inference.cpp pc_sd_upscaler.cpp pc_sd_nvfx_videoeffects.cpp)









