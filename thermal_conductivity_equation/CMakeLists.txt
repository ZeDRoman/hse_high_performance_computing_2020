cmake_minimum_required(VERSION 3.17)

project(tasks)
include_directories("/usr/include/mpi")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(tasks main.cpp)
