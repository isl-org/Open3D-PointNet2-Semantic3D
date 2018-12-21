# CMake 3.8+ has build-in cuda support
# ref: https://devblogs.nvidia.com/building-cuda-applications-cmake/
# ref: https://stackoverflow.com/a/51769885/1255535
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tf_ops LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Open3D cmake
include(open3d.cmake)
message(STATUS "Open3D installed to: ${Open3D_LIBRARY_DIRS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS} -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")

# Custom finders path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

# This exports the following variables:
# TensorFlow_INCLUDE_DIR
# TensorFlow_DIR
# TensorFlow_CXX_ABI
# TensorFlow_GIT_VERSION
# TensorFlow_VERSION
find_package(TensorFlow REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}")

link_directories(
    ${TensorFlow_DIR}
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)
include_directories(
    ${TensorFlow_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Grouping
add_library(tf_grouping SHARED
    tf_grouping.cu
    tf_grouping.cpp
)
target_compile_features(tf_grouping PUBLIC cxx_std_11)
target_link_libraries(tf_grouping cudart tensorflow_framework)
set_target_properties(tf_grouping PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Interpolate
link_directories(${Open3D_LIBRARY_DIRS})
add_library(tf_interpolate SHARED
    tf_interpolate.cpp
)
target_compile_features(tf_interpolate PUBLIC cxx_std_11)
target_include_directories(tf_interpolate PUBLIC ${Open3D_INCLUDE_DIRS})
target_link_libraries(tf_interpolate tensorflow_framework ${Open3D_LIBRARIES})

# Grouping
add_library(tf_sampling SHARED
    tf_sampling.cu
    tf_sampling.cpp
)
target_compile_features(tf_sampling PUBLIC cxx_std_11)
target_link_libraries(tf_sampling cudart tensorflow_framework)
set_target_properties(tf_sampling PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
