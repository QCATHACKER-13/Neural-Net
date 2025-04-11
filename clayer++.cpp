#include <iostream>
#include <vector>
#include "clayer++.h"

using namespace std;

int main() {
    // Define input and target output
    vector<double> inputs = {0.5, 0.1, 0.4}; // Example single input
    vector<vector <double>> targets = {{1.0, 0.0}, {0.0, 1.0}, {1.0}}; // Expected outputs for each neuron

    double learning_rate = 1e-2;
    double decay_rate = 0.001;
    vector<double> beta = {0.9, 0.9999}; // Momentum factor for Adam
    int step_size = 10; // For step decay learning rate

    // Initialize Layer with Adam optimizer and Step Decay learning rate
    Layer vlayer(
        2, inputs, targets[0], 
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    Layer hlayer(
        2, vlayer.get_output(), targets[1], 
        learning_rate, decay_rate, beta,
        HIDDEN, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );

    Layer olayer(
        1, hlayer.get_output(), targets[2], 
        learning_rate, decay_rate, beta,
        OUTPUT, LEAKY_RELU, EXPDECAY, ADAM, MSE
    );
    
    vlayer.set_step_size(step_size);
    hlayer.set_step_size(step_size);
    olayer.set_step_size(step_size);

    //Perform feedforward operation
    vlayer.feedforward();
    vlayer.measurement();

    hlayer.feedforward();
    hlayer.measurement();

    olayer.feedforward();
    olayer.measurement();

    // Perform backpropagation
    vlayer.backpropagation();
    hlayer.backpropagation();
    olayer.backpropagation();

    //layer.training(1000, 0.01, true);

    return 0;
}