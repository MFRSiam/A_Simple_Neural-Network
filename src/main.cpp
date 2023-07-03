#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "Network.hpp"
#include "MlAlgorithms.hpp"

int main() {
    auto *n1 = new Network({4,3,3});

    std::vector<std::vector<double>> inputData = {
            {1.0,2.0,3.0,2.5},
            {2.0,5.0,-1.0,2.0},
            {-1.5, 2.7, 3.3, -0.8}
    };

//    std::vector<std::vector<double>> weights1 = {
//            {0.2,0.8,-0.5,1.0},
//            {0.5,-0.91,0.26,-0.5},
//            {-0.26,-0.27,0.17,0.87}
//    };
//    n1->setLayerWeights(weights1);

//    std::vector<std::vector<double>> weights2 = {
//            {0.1, -0.14, 0.5},
//            {-0.5, 0.12, -0.33},
//            {-0.44, 0.73, -0.13}
//    };
//    n1->setLayerWeights(weights2,2);

    n1->setRandomWeights();
    n1->setZeroBias();
    n1->setDenseLayerActivation(MlAlgorithm::Activations::Relu_Activation);
//    n1->setLayerBias({2.0,3.0,0.5});
//    n1->setLayerBias({-1, 2, -0.5},2);
    for(const auto & i : inputData){
        n1->setLayer(i);
        n1->feedForward();
        auto temp =n1->getOutputLayer();
        for(auto x:*temp){
            std::printf("%lf ",x.value);
        }
        std::printf("\n");

    }

    delete n1;
    return EXIT_SUCCESS;
}