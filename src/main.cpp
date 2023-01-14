#include <iostream>
#include <vector>
#include <armadillo>

int main() {
    // Code a Layer of neurons
    arma::mat inputs = {1.0,2.0,3.0,2.5};
    arma::mat weights(3,4);
    weights = {
            {0.2,0.8,-0.5,1},
            {0.5,-0.91,0.26,-0.5},
            {-0.26,-0.27,0.17,0.87}
    };
    arma::mat biases = {2.0,3.0,0.5};
    arma::mat output(1,3);
    for(int i=0;i<3;i++){
        output(i) = arma::dot(inputs,weights.row(i)) + biases(i);
    }
    std::cout << output << "\n";
    return EXIT_SUCCESS;
}