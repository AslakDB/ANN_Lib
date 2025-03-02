#include "Neuron.h"
#include <random>



Neuron::Neuron(int numInputs)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    learningRate = 0.1;
    bias = dist(gen);
    for (int i = 0; i < numInputs; i++)
    {
        weights.push_back(dist(gen));
    }
}
