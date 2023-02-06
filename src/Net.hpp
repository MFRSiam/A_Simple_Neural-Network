#ifndef NEURALNETWORK_PROJECT_NET_HPP
#define NEURALNETWORK_PROJECT_NET_HPP
#include <vector>

/*!
 * @brief This Namespace contains code that i implemented myself without following any books or tutorial
 * Please Be Careful while using this namespace
 */
namespace NeuralNet {


    struct Connection{
        double weight;
        double deltaWeight;
        Connection();
    };



    class Neuron{
    public:
        Neuron(unsigned numOutputs);

    private:
        double p_outputValue;
        std::vector<Connection> p_outputWeights;
    };


    typedef std::vector<Neuron> Layers;


    class Net {
    public:
        Net(const std::vector<uint32_t> &topology);
        void FeedForward(const std::vector<double> &inputVals);
        void BackProp(const std::vector<double> &targetVals);
        void GetResult(std::vector<double> &resultVals) const;

    private:
        std::vector<Layers> p_Layers;
    };

} // NeuralNet

#endif
