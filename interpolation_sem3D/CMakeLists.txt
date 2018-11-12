cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11")
project(interpolation)

add_executable(interpolate main.cpp)
target_include_directories(interpolate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)

add_executable(eigen_print eigen_print.cpp)
target_include_directories(eigen_print PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)
