#pragma once
#include "Neuron.h"
class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int>& topology);
    void train(const std::vector<std::vector<double>>& trainingInputs,
               const std::vector<std::vector<double>>& trainingTargets,
               int epochs,
               double targetError = 0.01);
    std::vector<double> predict(const std::vector<double>& inputs);

    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    static double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

private:
    std::vector<std::vector<Neuron>> layers;
    double errorRate(std::vector<double>& expected, std::vector<double>& actual);
    std::vector<double> feedForward(const std::vector<double>& inputs);
    void backPropagate(const std::vector<double>& expected);
};
