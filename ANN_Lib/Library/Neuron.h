#pragma once
#include <iostream>
#include <vector>
#include <cmath>

class Neuron
{
    
public:
   
    std::vector<double> weights;
    double bias;
    double output;
    double error;
    double learningRate;

    
    Neuron( int numInputs);
};
