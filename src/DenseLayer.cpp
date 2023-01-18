//
// Created by M.F.R.Siam on 1/17/2023.
//

#include "DenseLayer.hpp"

DenseLayer::DenseLayer(int n_inputs, int n_neurons) {
    // Todo: Initialize Weights and biases
    this->weights = 0.1 * arma::randu(n_inputs,n_neurons);
    this->biases = arma::zeros(1,n_neurons);
}

void DenseLayer::forward(arma::mat inputs) {
    // Todo: Calculate Output Valuse from inputs, weights and biases

}

arma::mat &DenseLayer::getWeight() {
    return this->weights;
}

arma::mat &DenseLayer::getBiases() {
    return this->biases;
}
