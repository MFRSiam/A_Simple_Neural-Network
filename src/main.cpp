#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "Network.hpp"

int main() {
    Network *n1 = new Network({4,3});

    n1->setLayer({1.0,2.0,3.0,2.5});
    std::vector<std::vector<double>> weights = {
            {0.2,0.8,-0.5,1.0},
            {0.5,-0.91,0.26,-0.5},
            {-0.26,-0.27,0.17,0.87}
    };
    n1->setLayerWeights(weights);
    n1->setLayerBias({2.0,3.0,0.5});
    return EXIT_SUCCESS;
}