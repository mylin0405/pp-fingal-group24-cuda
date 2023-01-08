cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

# Find the CUDA package
find_package(CUDA REQUIRED)
message(CHECK_START "Finding CUDA...")
if(CUDA_FOUND)
    message(CHECK_PASS "Found CUDA.")
else()
    message(CHECK_FAIL "CUDA not found.")
endif()

# Set the host compiler for the CUDA code
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# add GP_CUDA library
add_library(GP_CUDA cuda_utils.cu)

# specificy include dirs for the GP_CUDA target
target_include_directories(
    GP_CUDA
    PRIVATE ${CUDA_INCLUDE_DIRS}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp
)

# Set the target's language to CUDA
set_target_properties(GP_CUDA PROPERTIES
    LANGUAGE CUDA
)

# Set computability 70 75 80
set_target_properties(GP_CUDA PROPERTIES
    CUDA_ARCHITECTURES 70 75 80
)
