//
// Created by mfrfo on 2/5/2023.
//

#include <random>
#include <cassert>
#include "Net.hpp"

namespace NeuralNet {

    Connection::Connection() {
        srand(time(nullptr));
        weight = (double) rand() / RAND_MAX;
    }

    /*
     * -----------------------------------------------------------------------
     * Neuron Class
     * -----------------------------------------------------------------------
     */

    Neuron::Neuron(unsigned int numOutputs) {
        for(int c=0;c<numOutputs;c++){
            p_outputWeights.emplace_back(Connection());
        }
    }


    /*
     * ------------------------------------------------------------------------
     * NET CLASS
     * ------------------------------------------------------------------------
     */

    Net::Net(const std::vector<uint32_t> &topology) {
        // Numbers of Layers Should Be Passed In the topology
        unsigned numLayers = topology.size();
        for(unsigned layerNumber = 0;layerNumber<numLayers;layerNumber++){
            unsigned numOutputs = layerNumber == topology.size() - 1 ? 0 : topology[layerNumber + 1];

            for(unsigned neuronNum = 0; neuronNum <= topology[layerNumber]; neuronNum++){
                p_Layers.back().emplace_back(Neuron(numOutputs));
            }
        }
    }

    void Net::FeedForward(const std::vector<double> &inputVals) {
        assert(inputVals.size() == p_Layers[0].size()-1);

    }

    void Net::BackProp(const std::vector<double> &targetVals) {

    }

    void Net::GetResult(std::vector<double> &resultVals) const {

    }

} // NeuralNet