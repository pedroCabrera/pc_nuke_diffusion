cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4101")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4018")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4305")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4715")

endif()

add_subdirectory(packages/stable-diffusion.cpp)

project(pc_nuke_diffusion LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH "CMake;${CMAKE_MODULE_PATH}")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

add_definitions(-DSD_CUDA)
add_definitions(-DSD_USE_CUDA)

find_package(Nuke REQUIRED)



function(add_plugin PLUGIN_NAME)
    add_nuke_plugin(${PLUGIN_NAME} ${ARGN})
    target_compile_definitions(${PLUGIN_NAME} PRIVATE FN_EXAMPLE_PLUGIN)
    # Link the stable-diffusion static library
    target_link_libraries(${PLUGIN_NAME}  PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${PLUGIN_NAME}  PUBLIC c_std_11 cxx_std_17)    
endfunction()

add_plugin(pc_sd_load_model pc_sd_load_model.cpp)
add_plugin(pc_sd_inference pc_sd_inference.cpp)
add_plugin(pc_sd_upscaler pc_sd_upscaler.cpp)
set(TARGET sd)

add_executable(${TARGET} "packages/stable-diffusion.cpp/examples/cli/main.cpp")
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

