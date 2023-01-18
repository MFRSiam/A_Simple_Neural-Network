#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
// Todo: Learn About Armadillo
#include <armadillo>
// Todo: Learn Matplot++
#include <matplot/matplot.h>

// My Own Imports
#include "DenseLayer.hpp"
#include "Helper.hpp"


int main() {
    arma::mat random(2,5,arma::fill::randu);
    arma::mat zeros(2,5,arma::fill::zeros);
    DenseLayer dl(2,5);
    std::cout << dl.getBiases() << std::endl;
    std::cout << dl.getWeight() << std::endl;
    GeneratedData G = ReadData();
    std::cout << G.X << std::endl;
    std::cout << G.y << std::endl;
    return EXIT_SUCCESS;
}