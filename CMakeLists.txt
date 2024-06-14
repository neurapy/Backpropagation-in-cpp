cmake_minimum_required(VERSION 3.10)

project(NeuralNetFromScratch VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(NeuralNetFromScratch main.cpp neural_network.cpp dataset.cpp)

target_include_directories(NeuralNetFromScratch PUBLIC "${PROJECT_SOURCE_DIR}")
