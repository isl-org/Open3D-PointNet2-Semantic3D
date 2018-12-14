include(ProcessorCount)
ProcessorCount(num_cores)

configure_file(${CMAKE_SOURCE_DIR}/open3d_builder.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/open3d/CMakeLists.txt
               COPYONLY)

execute_process(
    COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR} .
                             -DCMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}
                             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                             -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/open3d)
execute_process(
    COMMAND ${CMAKE_COMMAND} --build . -- -j ${num_cores}
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/open3d")
set(open3d_root ${CMAKE_CURRENT_BINARY_DIR}/open3d)
