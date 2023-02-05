//
// Created by mfrfo on 2/5/2023.
//

#include "Net.hpp"

namespace NeuralNet {

    Net::Net(const std::vector<uint32_t> &topology) {
        // Numbers of Layers Should Be Passed In the topology
        unsigned numLayers = topology.size();
        for(unsigned layerNumber = 0;layerNumber<numLayers;layerNumber++){
            
        }
    }

    void Net::FeedForward(const std::vector<double> &inputVals) {

    }

    void Net::BackProp(const std::vector<double> &targetVals) {

    }

    void Net::GetResult(std::vector<double> &resultVals) const {

    }
} // NeuralNet