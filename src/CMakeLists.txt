cmake_minimum_required(VERSION 3.15)

project(NCC)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../)

add_executable(encode encode.cc)
add_executable(decode decode.cc)

target_link_libraries(encode dlib blas lapack)