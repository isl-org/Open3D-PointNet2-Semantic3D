cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tf_ops LANGUAGES CXX CUDA)
set (CMAKE_CXX_STANDARD 11)

add_library(tf_grouping_g STATIC tf_grouping_g.cu)
target_compile_features(tf_grouping_g PUBLIC cxx_std_11)
set_target_properties(tf_grouping_g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
