cmake_minimum_required(VERSION 3.20)
project(NeuralNetwork_Project)

set(CMAKE_CXX_STANDARD 23)
find_package(Armadillo CONFIG REQUIRED)
find_package(matplot++ CONFIG REQUIRED)

#For Raylib
set(RAYLIB_VERSION 4.5.0)
find_package(raylib ${RAYLIB_VERSION} QUIET) # QUIET or REQUIRED
if (NOT raylib_FOUND) # If there's none, fetch and build raylib
    include(FetchContent)
    FetchContent_Declare(
            raylib
            DOWNLOAD_EXTRACT_TIMESTAMP OFF
            URL https://github.com/raysan5/raylib/archive/refs/tags/${RAYLIB_VERSION}.tar.gz
    )
    FetchContent_GetProperties(raylib)
    if (NOT raylib_POPULATED) # Have we downloaded raylib yet?
        set(FETCHCONTENT_QUIET NO)
        FetchContent_Populate(raylib)
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
        add_subdirectory(${raylib_SOURCE_DIR} ${raylib_BINARY_DIR})
    endif()
endif()









# Main Project
add_executable(NeuralNetwork_Project src/main.cpp src/Helper.hpp src/Network.cpp src/Network.hpp src/Gui.cpp src/Gui.hpp src/MlAlgorithms.cpp src/MlAlgorithms.hpp)

if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

#Linkers
target_include_directories(NeuralNetwork_Project PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(NeuralNetwork_Project PUBLIC ${ARMADILLO_LIBRARIES})
target_link_libraries(NeuralNetwork_Project PUBLIC Matplot++::matplot)
target_link_libraries(NeuralNetwork_Project PUBLIC raylib)