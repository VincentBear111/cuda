cmake_minimum_required(VERSION 3.21)
set(MyProject add)
project(${MyProject})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(INC_DIR /usr/local/cuda-11.7/include)
set(LIB_DIR /usr/local/cuda-11.7/lib64)


include_directories(${INC_DIR})
link_directories(${LIB_DIR})

link_libraries()

add_executable(${MyProject} 
            add.cu
            add.cpp
            )