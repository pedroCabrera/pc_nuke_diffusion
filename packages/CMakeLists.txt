cmake_minimum_required(VERSION 3.10)
project(pc_deps LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DSD_CUDA -DSD_USE_CUDA)

if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/bigobj>
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:C>:/MP>
    $<$<COMPILE_LANGUAGE:CXX>:/MP>)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/bigobj /MP\"")
  
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)

set(MAXINE_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Maxine-VFX-SDK")

add_library(nvvfx_proxy STATIC
  "${MAXINE_SDK_DIR}/nvvfx/src/nvVideoEffectsProxy.cpp"
  "${MAXINE_SDK_DIR}/nvvfx/src/nvCVImageProxy.cpp")
target_include_directories(nvvfx_proxy PUBLIC "${MAXINE_SDK_DIR}/nvvfx/include")
target_compile_features(nvvfx_proxy PUBLIC c_std_11 cxx_std_17)

include(GNUInstallDirs)

install(TARGETS stable-diffusion ggml zip nvvfx_proxy
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  
)
install(DIRECTORY ${MAXINE_SDK_DIR}/nvvfx/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/ggml/examples/stb_image.h
              ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/ggml/examples/stb_image_write.h
              ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty/stb_image_resize.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ---------------------------------------------------------------------------------
# CLI EXE for testing

set(PLUGIN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nuke/pc_nuke_diffusion")
set(TARGET sd)
add_executable(${TARGET} "${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/examples/cli/main.cpp")
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion  ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)
install(TARGETS sd
  RUNTIME DESTINATION ${PLUGIN_OUTPUT_DIR}
)
# ---------------------------------------------------------------------------------        
