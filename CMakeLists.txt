cmake_minimum_required(VERSION 3.16)
# Language == C, CXX, CUDA
project(pp_final LANGUAGES C CXX CUDA)
# CXX 20
set(CMAKE_CXX_STANDARD 20)
# -O3
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set CUDA HOST COMPILER
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

add_subdirectory(include/GP)
add_subdirectory(include/cuda)

add_executable(app
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(app
    PRIVATE GP GP_CUDA
)
set_target_properties(app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_HOME_DIRECTORY}/bin"
)
