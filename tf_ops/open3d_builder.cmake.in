cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(open3d_builder NONE)

include(ExternalProject)

ExternalProject_Add(
    open3d
    GIT_REPOSITORY https://github.com/IntelVCL/Open3D.git
    GIT_TAG 33e46f7 # Thu Nov 29 21:19:44
    PREFIX open3d
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DWITH_OPENMP=ON
               -DBUILD_EIGEN3=ON
               -DBUILD_GLEW=ON
               -DBUILD_GLFW=ON
               -DBUILD_JPEG=ON
               -DBUILD_JSONCPP=ON
               -DBUILD_PNG=ON
               -DBUILD_TINYFILEDIALOGS=ON
               -DBUILD_CPP_EXAMPLES=ON
               -DBUILD_PYBIND11=OFF
               -DBUILD_PYTHON_MODULE=OFF
               -DENABLE_JUPYTER=OFF
               -DBUILD_PYTHON_TUTORIALS=OFF
)
