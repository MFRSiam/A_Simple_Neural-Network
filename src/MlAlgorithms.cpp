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
} // MlAlgorithm