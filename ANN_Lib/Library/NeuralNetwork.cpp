#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int>& topology)
{
    if (topology.size() < 2)
        throw std::invalid_argument("Topology must have at least 2 layers.");

    // Create the layers
    for (size_t i = 0; i < topology.size(); i++)
    {
        std::vector<Neuron> layer;
        int numNeurons = topology[i];
        int numInputs = (i == 0) ? 0 : topology[i - 1];
        for (int j = 0; j < numNeurons; j++)
        {
            layer.push_back(Neuron(numInputs));
        }
        layers.push_back(layer);
    }
}

double NeuralNetwork::errorRate(std::vector<double>& expected, std::vector<double>& actual)
{
    double error = 0;
    for (int i = 0; i < expected.size(); i++)
    {
        error += pow(expected[i] - actual[i], 2);
    }
    return error;
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs)
{
    if (inputs.size() != layers[0].size())
        throw std::invalid_argument("Input size does not match network input layer size.");

    // Set initial outputs as the network's inputs
    std::vector<double> outputs = inputs;

    // Process through each layer starting from the first hidden layer.
    for (size_t i = 1; i < layers.size(); i++)
    {
        std::vector<double> newOutputs;
        for(auto& neuron : layers[i])
        {
            double activation = neuron.bias;
            for (size_t j = 0; j < outputs.size(); j++)
                activation += outputs[j] * neuron.weights[j];

            // Apply the sigmoid activation function.
            neuron.output = sigmoid(activation);
            newOutputs.push_back(neuron.output);
        }
        outputs = newOutputs;
    }
    return outputs;
}

void NeuralNetwork::backPropagate(const std::vector<double>& expected)
{
    // Calculate output layer errors
    size_t outputLayer = layers.size() - 1;
    for (size_t i = 0; i < layers[outputLayer].size(); i++)
    {
        Neuron& neuron = layers[outputLayer][i];
        double output = neuron.output;
        // Error delta = (target - output) * output * (1 - output)
        neuron.error = (expected[i] - output) * sigmoidDerivative(output);
    }

    // Calculate hidden layer errors
    for (int layerIdx = outputLayer - 1; layerIdx > 0; layerIdx--)
    {
        for (size_t i = 0; i < layers[layerIdx].size(); i++)
        {
            double error = 0.0;
            // Sum the errors from the next layer
            for (size_t j = 0; j < layers[layerIdx + 1].size(); j++)
            {
                error += layers[layerIdx + 1][j].error * layers[layerIdx + 1][j].weights[i];
            }
            Neuron& neuron = layers[layerIdx][i];
            // Error delta = error * output * (1 - output)
            neuron.error = error * sigmoidDerivative(neuron.output);
        }
    }

    // Update weights and biases
    for (size_t layerIdx = 1; layerIdx < layers.size(); layerIdx++)
    {
        const auto& prevLayer = layers[layerIdx - 1];
        for (auto& neuron : layers[layerIdx])
        {
            // Update bias
            neuron.bias += neuron.learningRate * neuron.error;
            // Update weights
            for (size_t i = 0; i < prevLayer.size(); i++)
            {
                neuron.weights[i] += neuron.learningRate * neuron.error * prevLayer[i].output;
            }
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& trainingInputs,
    const std::vector<std::vector<double>>& trainingTargets, int epochs, double targetError)
{
    
        if (trainingInputs.size() != trainingTargets.size())
            throw std::invalid_argument("Number of training inputs must match number of target outputs");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0.0;
        
            // Train on each input-target pair
            for (size_t i = 0; i < trainingInputs.size(); i++)
            {
                // Forward pass
                auto outputs = feedForward(trainingInputs[i]);
            
                // Calculate error
                totalError += errorRate(const_cast<std::vector<double>&>(trainingTargets[i]), outputs);
            
                // Backward pass
                backPropagate(trainingTargets[i]);
            }
        
            // Check if we've reached target error
            double averageError = totalError / trainingInputs.size();
            if (averageError <= targetError)
                break;
        }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs)
{
    return feedForward(inputs);
}
