//
// Created by M.F.R.Siam on 7/2/2023.
//

#ifndef NEURALNETWORK_PROJECT_NETWORK_HPP
#define NEURALNETWORK_PROJECT_NETWORK_HPP
#include <vector>
#include <string>
#include <random>

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

    void setWeights(const std::vector<std::vector<double>> &weights){
        if(m_weights.size()!=weights.size()){
            throw std::exception("Incompatible Size");
        }
        for(int i=0;i<weights.size();i++){
            m_weights[i] = weights[i];
        }
    }

    void setWeights(){
        const double mean = 0.0f;
        const double varience = 1.0f;
        std::random_device rd{};
        std::mt19937 engine{rd()};
        std::normal_distribution<double> dist(mean,varience);

        for(int i=0;i<m_weights.size();i++){
            for(int j=0;j<m_weights[i].size();j++){
                m_weights[i][j] = dist(engine);
            }
        }
    }

    void setBias(std::vector<double> biases){
        for(int i=0;i<m_layer.size();i++){
            m_layer[i].bias = biases[i];
        }
    }
    std::vector<Neuron> *getLayer(){
        return &m_layer;
    }
    std::vector<std::vector<double>> *getWeight(){
        return &m_weights;
    }

    int getSize(){
        return (int)m_layer.size();
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

    void setLayer(const std::vector<double> &layer){
        m_network[0].set(layer);
    }


    void setLayerWeights(const std::vector<std::vector<double>> &weights,int layerNum=1){
        m_network[layerNum].setWeights(weights);
    }

    void setRandomWeights(){
        for(int i=0;i<m_network.size();i++){
            m_network[i].setWeights();
        }
    }

    void setLayerBias(std::vector<double> &&biases,int layerNum = 1){
        m_network[layerNum].setBias(biases);
    }
    void feedForward(){
        for(int i=1;i<m_network.size();i++){
            auto val = m_network[i].getLayer();
            auto tempWeight = m_network[i].getWeight();
            int prevLayerSize = m_network[i-1].getSize();
            auto prevLayer = m_network[i-1].getLayer();
            for(int currentNeuron = 0; currentNeuron < val->size(); currentNeuron++){
                double temp = 0;
                for(int j=0;j<prevLayerSize;j++){
                    temp += (*prevLayer)[j].value * (*tempWeight)[currentNeuron][j];
                }
                temp += (*val)[currentNeuron].bias;

                (*val)[currentNeuron].value = temp;
            }
        }
    }
    std::vector<Neuron>* getOutputLayer(){
        return m_network[m_network.size()-1].getLayer();
    }

};


#endif //NEURALNETWORK_PROJECT_NETWORK_HPP
