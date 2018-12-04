# CMake 3.8+ has build-in cuda support
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tf_ops LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

link_directories(
    /home/ylao/repo/venvs/semantic3/lib/python3.6/site-packages/tensorflow
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)
include_directories(
    /home/ylao/repo/venvs/semantic3/lib/python3.6/site-packages/tensorflow/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(tf_grouping_so SHARED
    tf_grouping_g.cu
    tf_grouping.cpp
)
target_compile_features(tf_grouping_so PUBLIC cxx_std_11)
set_target_properties(tf_grouping_so PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tf_grouping_so cudart tensorflow_framework)
