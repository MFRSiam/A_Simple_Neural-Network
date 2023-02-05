cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork_Project)

set(CMAKE_CXX_STANDARD 23)
find_package(Armadillo CONFIG REQUIRED)
find_package(matplot++ CONFIG REQUIRED)

add_executable(NeuralNetwork_Project src/main.cpp src/DenseLayer.cpp src/DenseLayer.hpp src/Helper.hpp src/Net.cpp src/Net.hpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_link_libraries(NeuralNetwork_Project PUBLIC ${ARMADILLO_LIBRARIES})
target_link_libraries(NeuralNetwork_Project PUBLIC Matplot++::matplot)