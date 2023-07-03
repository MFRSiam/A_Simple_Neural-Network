//
// Created by M.F.R.Siam on 7/3/2023.
//

#include "MlAlgorithms.hpp"

namespace MlAlgorithm {
    double Activations::Relu_Activation(double num) {
        if(num > 0){
            return num;
        }else{
            return 0.0;
        }
    }

    void Activations::SoftMax_Activation(std::vector<Neuron> &outputs) {
        double sum = 0.0;
        for(auto x:outputs){
            sum += std::exp(x.value);
        }

        for(auto &x:outputs){
            x.value = std::exp(x.value)/sum;
        }

        // Checker Code Should Remove
        double sumTmp = 0.0;
        for(auto x:outputs){
            sumTmp += x.value;
        }
        if(sumTmp - 1.0 < 0.0005){
            std::cout << "SoftMAX Looks OK\n";
        }
    }
} // MlAlgorithm