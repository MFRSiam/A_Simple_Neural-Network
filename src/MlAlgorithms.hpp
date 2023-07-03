//
// Created by M.F.R.Siam on 7/3/2023.
//

#ifndef NEURALNETWORK_PROJECT_MLALGORITHMS_HPP
#define NEURALNETWORK_PROJECT_MLALGORITHMS_HPP
#include <cmath>
#include <vector>
#include <iostream>
#include "Network.hpp"

namespace MlAlgorithm {


    class Activations {
    public:
        static double Relu_Activation(double num);

        static void SoftMax_Activation(std::vector<Neuron>& outputs);
    };

} // MlAlgorithm

#endif //NEURALNETWORK_PROJECT_MLALGORITHMS_HPP
