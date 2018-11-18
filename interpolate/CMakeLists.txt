
cmake_minimum_required(VERSION 3.0.0)
if (NOT CMAKE_VERSION VERSION_LESS "3.1")
    cmake_policy(SET CMP0054 NEW)
endif()
set (CMAKE_CXX_STANDARD 11)

project(interpolation)

# Open3D
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
list(APPEND Open3D_LIBRARIES dl)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")
link_directories(${Open3D_LIBRARY_DIRS})
if (Open3D_FOUND)
    message(STATUS "Found Open3D at ${Open3D_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Open3D not found")
endif()

add_definitions(-DUNIX)
add_compile_options(-Wno-deprecated-declarations)
add_compile_options(-Wno-unused-result)
add_definitions(-O3)

# OpenMP
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

add_executable(interpolate interpolate.cpp)
target_link_libraries(interpolate ${Open3D_LIBRARIES})
target_include_directories(interpolate PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/Eigen
                           ${Open3D_INCLUDE_DIRS})
