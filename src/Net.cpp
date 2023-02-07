//
// Created by mfrfo on 2/5/2023.
//

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

    Neuron::Neuron(unsigned int numOutputs,unsigned myIndex) {
        for(int c=0;c<numOutputs;c++){
            p_outputWeights.emplace_back(Connection());
        }
        p_myIndex = myIndex;
    }

    void Neuron::setOutputValue(const double &value) {
        p_outputValue = value;
    }

    double Neuron::getOutputValue() const {
        return p_outputValue;
    }


    void Neuron::feedForward(const Layer &prevLayer) {
        double sum = 0.0;
        for(unsigned n=0;n<prevLayer.size();n++){
            sum += prevLayer[n].getOutputValue() * prevLayer[n].p_outputWeights[p_myIndex].weight;
        }
        p_outputValue = Neuron::transformFunction(sum);
    }

    double Neuron::transformFunction(double sum) {
        // tanh
        return std::tanh(sum);
    }

    double Neuron::transformFunctionDerivative(double sum) {
        return 1 - sum*sum;
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
                p_Layers.back().emplace_back(Neuron(numOutputs,neuronNum));
            }
        }
    }

    void Net::FeedForward(const std::vector<double> &inputVals) {
        assert(inputVals.size() == p_Layers[0].size()-1);
        for(unsigned i=0;i<inputVals.size();i++){
            p_Layers[0][i].setOutputValue(inputVals[i]);
        }

        for(unsigned layerNum = 1;layerNum < p_Layers.size();layerNum++){
            Layer &prevLayer = p_Layers[layerNum - 1];
            for(unsigned n=0;n < p_Layers[layerNum].size() - 1;n++){
                p_Layers[layerNum][n].feedForward(prevLayer);
            }
        }
    }

    void Net::BackProp(const std::vector<double> &targetVals) {
        // Calculate overall net error (RMS)

        // Calculate Output Layer Gradients

        // Calculate Gradients On Hidden Layer

        /*
         * For All Layers from output layer to the 1st hidden layer
         * update the connection weights
         */
    }

    void Net::GetResult(std::vector<double> &resultVals) const {

    }

} // NeuralNet