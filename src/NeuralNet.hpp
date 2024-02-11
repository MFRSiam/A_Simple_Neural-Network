//
// Created by mfrfo on 2/8/2024.
//

#ifndef NEURALNETWORK_PROJECT_NEURAL_NET_HPP
#define NEURALNETWORK_PROJECT_NEURAL_NET_HPP
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>

class Neural_Net{
public:
    explicit Neural_Net(std::vector<int> &&structure){
        net.resize(structure.size());

        for(int i=0;i<structure.size();i++){
            net[i].resize(structure[i]);
        }
        weights.resize(structure.size()-1);
        for(int i=0;i<weights.size();i++){
            weights[i].resize(net[i+1].size());
            for(int j=0;j<net[i+1].size();j++){
                weights[i][j].resize(net[i].size());
            }
        }

        bias.resize(structure.size()-1);
        for(int i=0;i<bias.size();i++){
            bias.resize(structure[i+1]);
        }

    }

    static float sigmoid(float sum){
        return (float)1.0/(float)(1.0+std::exp(-sum));
    }

    void feedForward(const std::vector<float> &inputData){
        if(inputData.size() != net[0].size()){
            throw std::exception("Input Data and Net Size didnt match");
        }
        //Input Layer
        net[0] = inputData;
        // Hidden Layer
        for(int i=1;i<net.size()-1;i++){
            for(int neuron=0;neuron<net[i].size();neuron++){
                const std::vector<std::vector<float>> &layer_weights = weights[i-1];
                float sum = weightsXactivations(net[i-1],layer_weights[neuron]);
                sum += bias[i-1][neuron];
                sum = sigmoid(sum);
                net[i][neuron] = sum;
            }
        }
        //Output Layer
        for(int neuron=0;neuron<net[net.size()-1].size();neuron++){
            const std::vector<std::vector<float>> &layer_weight = weights[weights.size()-1];
            float sum = weightsXactivations(net[net.size()-1],layer_weight[neuron]);
            sum += bias[net.size()-1][neuron];
            sum += sigmoid(sum);
            net[net.size()-1][neuron] = sum;
        }
    }


    void SGD(){

    }

private:
    std::vector<std::vector<float>> net;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> bias;

    static float weightsXactivations(const std::vector<float> &prevLayer,const std::vector<float> &weights){
        float sum = 0;
        for(int i=0;i<prevLayer.size();i++){
            sum += prevLayer[i] * weights[i];
        }
        return sum;
    }
};


#endif //NEURALNETWORK_PROJECT_NEURAL_NET_HPP
