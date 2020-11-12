cmake_minimum_required(VERSION 3.17)
project(tasks)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_STANDARD 17)

add_executable(tasks main.cpp)
