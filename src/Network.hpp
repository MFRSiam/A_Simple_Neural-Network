//
// Created by M.F.R.Siam on 7/2/2023.
//

#ifndef NEURALNETWORK_PROJECT_NETWORK_HPP
#define NEURALNETWORK_PROJECT_NETWORK_HPP
#include <vector>
#include <string>

struct Neuron{
    double value;
    double bias;
};

class Layer{
private:
    std::vector<Neuron> m_layer;
    std::vector<std::vector<double>> m_weights;
public:
    void resize(int size,int prevLayerSize = 0){
        m_layer.resize(size);
        if(prevLayerSize != 0){
            m_weights.resize(size);
            for(auto &x:m_weights){
                x.resize(prevLayerSize);
            }
        }
    }
    void set(const std::vector<double> &data){
        if(data.size() != m_layer.size()){
            throw std::exception("Incompatible Size");
        }
        for(int i=0;i<data.size();i++){
            m_layer[i].value = data[i];
        }
    }

    void setWeights(std::vector<std::vector<double>> &weights){
        if(m_weights.size()!=weights.size()){
            throw std::exception("Incompatible Size");
        }
        for(int i=0;i<weights.size();i++){
            m_weights[i] = weights[i];
        }
    }

    void setBias(std::vector<double> biases){
        for(int i=0;i<m_layer.size();i++){
            m_layer[i].bias = biases[i];
        }
    }
};


class Network {
private:
    std::vector<Layer> m_network;
public:
    explicit Network(std::vector<int> &&network){
        m_network.resize(network.size());
        for(int i=0;i<network.size();i++){
            m_network[i].resize(network[i]);
            if(i!=0){
                m_network[i].resize(network[i],network[i-1]);
            }
        }
    }

    void setLayer(std::vector<double> &&layer){
        m_network[0].set(layer);
    }

    void setLayerWeights(std::vector<std::vector<double>> &weights){
        m_network[1].setWeights(weights);
    }

    void setLayerBias(std::vector<double> &&biases){
        m_network[1].setBias(biases);
    }

};


#endif //NEURALNETWORK_PROJECT_NETWORK_HPP
