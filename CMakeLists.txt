cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork_Project)

set(CMAKE_CXX_STANDARD 20)
find_package(Armadillo CONFIG REQUIRED)
add_executable(NeuralNetwork_Project src/main.cpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

target_link_libraries(NeuralNetwork_Project PRIVATE ${ARMADILLO_LIBRARIES})
