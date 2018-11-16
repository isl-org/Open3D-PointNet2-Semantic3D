
cmake_minimum_required(VERSION 3.0.0)
if (NOT CMAKE_VERSION VERSION_LESS "3.1")
    cmake_policy(SET CMP0054 NEW)
endif()

project(interpolation)

find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    message(STATUS "Using installed OpenMP ${OpenMP_VERSION}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "OpenMP NOT found")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++11")

add_executable(interpolate main.cpp)
target_include_directories(interpolate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Eigen)
