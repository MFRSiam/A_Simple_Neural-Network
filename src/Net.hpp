#ifndef NEURALNETWORK_PROJECT_NET_HPP
#define NEURALNETWORK_PROJECT_NET_HPP
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
/*!
 * @brief This Namespace contains code that i implemented myself without following any books or tutorial
 * Please Be Careful while using this namespace
 */
namespace NeuralNet {
    class Neuron;
    typedef std::vector<Neuron> Layer;

    struct Connection{
        double weight;
        double deltaWeight;
        Connection();
    };


    class Neuron{
    public:
        Neuron(unsigned numOutputs,unsigned myIndex);
        void feedForward(const Layer &prevLayer);

        void setOutputValue(const double &value);
        double getOutputValue() const;

    private:
        static double transformFunction(double sum);
        static double transformFunctionDerivative(double sum);
        double p_outputValue;
        std::vector<Connection> p_outputWeights;
        unsigned p_myIndex;
    };



    class Net {
    public:
        Net(const std::vector<uint32_t> &topology);
        void FeedForward(const std::vector<double> &inputVals);
        void BackProp(const std::vector<double> &targetVals);
        void GetResult(std::vector<double> &resultVals) const;

    private:
        std::vector<Layer> p_Layers;
    };

} // NeuralNet

#endif
