/* Project: NEURA 
NEURA is an artificial neuron and network development research project 
focused on artificial intelligence for data analysis. The goal is to find 
the optimal neural network structure and improve artificial neurons for 
efficiency and speed.

Future projects:
- Project Mei: Neuron investigation project
- Project Raiden: Network development 
- Hardware integration: After successfully developing the artificial 
  neuron network, the project will transition into hardware implementation.

Developer: QCAT FERMI
*/


#ifndef CLAYER_H
#define CLAYER_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "cneuron++.h"

using namespace std;

// Enum for layer types
enum nltype { DEFAULT, HIDDEN, OUTPUT };

class Layer{
    private:
        // Layer properties
        int step_size, timestep = 0; // Training step parameters
        double decay_rate; // Decay rate for learning rate
        nltype layertype; // Layer type (input, hidden, output)
        
        // Layer parameters
        vector<Neuron> neuron;
        vector<double> input, bias, output, target, error, prediction, learning_rate, beta, probability, loss;// Weights, inputs, output, bias
        vector<vector<double>> weight; // Weight storage for neurons
        
        actfunc actFunc; // Activation function type
        lrs lr_schedule; // Learning rate adjustment strategy
        optimizer opt; // Optimization algorithm
        lossfunc lossFunc;

    public:
        // Constructor for initializing neural layer properties
        Layer (size_t num_neuron,
            const vector<double>& inputs, const vector<double>& targets, 
            const double& learning_rate, const double& decay_rate, const vector<double>& beta,
            nltype layertype, actfunc actfunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc)
        
        : input(inputs), target(targets), 
        learning_rate(learning_rate), decay_rate(decay_rate), beta(beta),
        actFunc(actfunc), lr_schedule(lr_schedule), opt(opt), lossFunc(lossFunc)
        
        {
            weight.resize(num_neuron);
            bias.resize(num_neuron);
            output.resize(num_neuron);
            error.resize(num_neuron);
            loss.resize(num_neuron);
            
            for(size_t i = 0; i < num_neuron; ++i){
                neuron.emplace_back(
                    Neuron(
                        inputs, targets[i], 
                        learning_rate, decay_rate, beta,
                        actfunc, lr_schedule, opt, lossFunc
                    )
                );
            }
            
            // Initialize weights and biases for each neuron
            for(size_t i = 0; i < num_neuron; ++i){
                neuron[i].initialize();
            }

            // Initialize weights and biases for the layer
            for(size_t i = 0; i < num_neuron; i++){
                weight[i].resize(input.size());            
                for(size_t j = 0; j < input.size(); ++j){
                    weight[i][j] = neuron[i].get_weight()[j];
                }
                bias[i] = neuron[i].get_bias();
            }
        }

        void feedforward() {
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].feedforward();           
                for(size_t j = 0; j < input.size(); ++j){
                    weight[i][j] = neuron[i].get_weight()[j];
                }
                output[i] = neuron[i].get_output();
                bias[i] = neuron[i].get_bias();
                error[i] = neuron[i].get_error();
            }
        }

        void softmax() {
            double maxVal = *max_element(output.begin(), output.end()); // Prevents overflow
            vector<double> exp_values(prediction.size());
        
            // Compute exponentials
            for (size_t i = 0; i < output.size(); ++i) {
                exp_values[i] = exp(output[i] - maxVal);
            }
        
            double sum_exp = accumulate(exp_values.begin(), exp_values.end(), 0.0);
        
            probability.resize(output.size());
        
            // Compute softmax probabilities
            for (size_t i = 0; i < output.size(); ++i) {
                if (sum_exp != 0) {
                    probability[i] = exp_values[i] / sum_exp;
                } else if (sum_exp == 0) {
                    probability[i] = 0.0;
                }
            }
        }

        // Loss functions and their derivatives
        void loss_function(const vector<double> prob_target) {
            for(size_t i = 0; i < prediction.size(); i++){
                switch (lossFunc) {
                    case MSE:
                        loss[i] = 0.5 * pow(prob_target[i] -  prediction[i], 2); // Mean squared error
                        break;
                        
                    case BCE:
                        //Binary cross entropy log() in C++ is base e or natural log ln()
                        loss[i] = - ((prob_target[i] * log(prediction[i])) + ((1 - prob_target[i]) * log(1 - prediction[i])))/prediction.size();
                        break;
                        
                    case HUBER:
                        double delta = 0.05; // Huber loss delta
                        if (abs(prob_target[i] - prediction[i]) <= delta) {
                            loss[i] =  0.5 * pow(prob_target[i] - prediction[i], 2);
                            break;
                        } else {
                            loss[i] =  delta * (abs(prob_target[i] - prediction[i]) - 0.5 * delta);
                            break;
                        }
                }
            }
        }
        

        void backpropagation(){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].backward();           
                for(size_t j = 0; j < input.size(); ++j){
                    weight[i][j] = neuron[i].get_weight()[j];
                }
                output[i] = neuron[i].get_output();
                bias[i] = neuron[i].get_bias();
            }
        }

        void measurement(){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].print_neuron(i);
            }
            cout << endl;            
        }

        vector <double> get_input() {return input;}
        vector<double> get_output(){return output;}
        vector <vector<double>> get_weight(){return weight;}
        vector<double> get_bias(){return bias;}
        vector<double> get_loss(){return loss;}
        vector<double> get_error(){return error;}
        vector<Neuron>& get_neuron() {return neuron;}        

        void set_bias(const vector<double>& biases){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_bias(biases[i]);
            }
        }
        void set_weight(const vector<vector<double>>& weights){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_weight(weights[i]);
            }
        }

        void set_error(const vector<double>& errors){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_error(errors[i]);
            }
        }

        void set_step_size(int stepsize){
            for(size_t i = 0; i < neuron.size(); i++){
                neuron[i].set_step_size(stepsize);
            }
        }
};

#endif