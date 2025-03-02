#include <iostream>
#include "Library/NeuralNetwork.h"

void testXOR()
{
    // Create XOR network with 2 inputs, 4 hidden neurons, and 1 output
    std::vector<int> topology = {2, 4,4, 1};
    NeuralNetwork network(topology);

    // XOR training data
    std::vector<std::vector<double>> trainingInputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

std::vector<std::vector<double>> trainingTargets = {
    {0.0},
    {1.0},
    {1.0},
    {0.0}
};

// Train the network
std::cout << "Training network...\n";
network.train(trainingInputs, trainingTargets, 1000000, 0.01);

// Test the network
std::cout << "\nTesting XOR gate:\n";
for (size_t i = 0; i < trainingInputs.size(); i++)
{
    auto result = network.predict(trainingInputs[i]);
    std::cout << trainingInputs[i][0] << " XOR " << trainingInputs[i][1]
              << " = " << result[0]
              << " (expected " << trainingTargets[i][0] << ")\n";
}
}

int main(int argc, char* argv[])
{
    try {
        testXOR();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}