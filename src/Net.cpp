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

    void Neuron::calcOutputGradient(double targetVal) {
        double delta = targetVal - p_outputValue;
        p_gradient = delta * Neuron::transformFunctionDerivative(p_outputValue);
    }


    void Neuron::calcHiddenGradient(const Layer &nextLayer) {
        double dow = sumDOW(nextLayer);
        p_gradient = dow * Neuron::transformFunctionDerivative(p_outputValue);
    }

    double Neuron::sumDOW(const Layer &nextLayer) const {
        double sum = 0.0;
        for(unsigned n=0;n<nextLayer.size()-1;n++){
            sum += p_outputWeights[n].weight * nextLayer[n].p_gradient;
        }
        return sum;
    }

    void Neuron::updateInputWeight(Layer &prevLayer) {
        for (unsigned n = 0; n < prevLayer.size(); ++n) {
            Neuron &neuron = prevLayer[n];
            double oldDeltaWeight = neuron.p_outputWeights[p_myIndex].deltaWeight;

            double newDeltaWeight = neuron.p_outputWeights[p_myIndex].deltaWeight = newDeltaWeight;
            neuron.p_outputWeights[p_myIndex].weight += newDeltaWeight;
        }
    }

    double Neuron::transformFunction(double sum) {
        // tanh
        return std::tanh(sum);
    }

    double Neuron::transformFunctionDerivative(double sum) {
        return 1 - sum*sum;
    }

    double Neuron::eta = 0.15;
    double Neuron::alpha = 0.5;


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
            p_Layers.back().back().setOutputValue(1.0);
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
        Layer &outputLayer = p_Layers.back();
        p_error = 0.0;

        for(unsigned n=0;n<outputLayer.size()-1;n++){
            double delta = targetVals[n] - outputLayer[n].getOutputValue();
            p_error += delta * delta;
        }
        p_error /= outputLayer.size() -1;
        p_error = std::sqrt(p_error);

        // Implement a Recent Average measurements
        p_recentAvgError = (p_recentAvgError * p_recentAvgSmootingFactor + p_error) / (p_recentAvgSmootingFactor + 1.0);
        // Calculate Output Layer Gradients

        for(unsigned n=0;n<outputLayer.size() -1;n++){
            outputLayer[n].calcOutputGradient(targetVals[n]);
        }

        // Calculate Gradients On Hidden Layer

        for(unsigned layerNum = p_Layers.size() -2; layerNum > 0; layerNum--){
            Layer &hiddenLayer = p_Layers[layerNum];
            Layer &nextLayer = p_Layers[layerNum+1];
            for(unsigned n=0;n<hiddenLayer.size();n++){
                hiddenLayer[n].calcHiddenGradient(nextLayer);
            }
        }

        /*
         * For All Layers from output layer to the 1st hidden layer
         * update the connection weights
         */
        for(unsigned layerNum = p_Layers.size()-1;layerNum > 0 ;layerNum--){
            Layer &layer = p_Layers[layerNum];
            Layer &prevLayer = p_Layers[layerNum-1];
            for(unsigned n=0;n<layer.size();n++){
                layer[n].updateInputWeight(prevLayer);
            }
        }
    }

    void Net::GetResult(std::vector<double> &resultVals) const {
        resultVals.clear();

        for (unsigned n = 0; n < p_Layers.back().size() - 1; ++n) {
            resultVals.push_back(p_Layers.back()[n].getOutputValue());
        }
    }

} // NeuralNet