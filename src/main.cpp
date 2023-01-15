#include <iostream>
#include <vector>
// Todo: Learn About Armadillo
#include <armadillo>

int main() {
    // Code a Layer of neurons
    arma::mat inputs = {
            {1.0,2.0,3.0,2.5},
            {2.0,5.0,-1.0,2.0},
            {-1.5,2.7,3.3,-0.8}
    };

    arma::mat weights = {
            {0.2,0.5,-0.26},
            {0.8,-0.91,-0.27},
            {-0.5,0.26,0.17},
            {1.0,-0.5,0.87}
    };

    arma::mat biases {2.0,3.0,0.5};
    arma::mat output(3,3);

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            output(i,j) = arma::dot(inputs.row(i),weights.col(j));
        }
        output.row(i) += biases;
    }
    std::cout << output << std::endl;

    return EXIT_SUCCESS;
}