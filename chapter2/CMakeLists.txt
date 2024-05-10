# required cmake version
cmake_minimum_required(VERSION 3.17)
set(MyProject hello) 
project(${MyProject})

# packages
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60;-G;-g)

file(GLOB_RECURSE   CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB           CURRENT_SOURCES  *.cpp *.cu)

source_group("INC" FILES ${CURRENT_HEADERS}) 
source_group("SRC" FILES ${CURRENT_SOURCES}) 

set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++17")

CUDA_ADD_EXECUTABLE(${MyProject} ${CURRENT_HEADERS} ${CURRENT_SOURCES})