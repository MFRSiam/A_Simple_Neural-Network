#include <iostream>
#include <vector>
// Todo: Learn About Armadillo
#include <armadillo>
// Todo: Learn Matplot++
#include <matplot/matplot.h>



int main() {
    arma::mat random(2,5,arma::fill::randu);
    arma::mat zeros(2,5,arma::fill::zeros);
    std::cout << random << std::endl;
    std::cout << zeros << std::endl;
    return EXIT_SUCCESS;
}