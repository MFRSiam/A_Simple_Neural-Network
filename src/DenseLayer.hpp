//
// Created by M.F.R.Siam on 1/17/2023.
//

#ifndef NEURALNETWORK_PROJECT_DENSELAYER_HPP
#define NEURALNETWORK_PROJECT_DENSELAYER_HPP
#include <armadillo>

/*!
 * Dense Layer
 */
class DenseLayer {
public:
    // Layer Initialization
    DenseLayer(int n_inputs, int n_neurons);

    // Forward Pass
    void forward(arma::mat inputs);

    arma::mat& getWeight();
    arma::mat& getBiases();
private:
    arma::mat weights;
    arma::mat biases;
};


#endif //NEURALNETWORK_PROJECT_DENSELAYER_HPP
