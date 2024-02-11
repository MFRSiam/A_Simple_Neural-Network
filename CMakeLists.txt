cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork_Project)

set(CMAKE_CXX_STANDARD 23)


# Main Project
add_executable(NeuralNetwork_Project src/main.cpp
        src/NeuralNet.cpp
        src/NeuralNet.hpp
)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()
