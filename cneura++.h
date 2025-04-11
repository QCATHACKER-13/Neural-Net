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

Developer: Christopher Emmanuelle J. Visperas, Applied Physics Researcher*/

#ifndef CNEURA_H
#define CNEURA_H

#include <iostream>
#include <vector>
#include <memory> // For std::unique_ptr
#include <algorithm>
#include <cmath>
#include <numeric>
#include "clayer++.h"

using namespace std;

class Neural{
    private:
        /*vector <Layer> layer;
        vector<vector<vector <double>>> weight;
        vector <vector <double>> output, bias, target, error, prediction, learning_rate;*/

        vector<unique_ptr<Layer>> layers; // Use unique_ptr for better memory management
        vector<vector<vector<double>>> weight; // Stores weights for all layers

    public:
        Neural(const vector <size_t>& num_neurons,
            const vector <double>& inputs, const vector<vector<double>>& targets,
            const double& learning_rate, const double& decay_rate, const vector<double>& beta,
            actfunc actFunc, lrs lr_schedule, optimizer opt, lossfunc lossFunc){
                
                //Intialized the neural layers
                layers.emplace_back(
                    make_unique<Layer>(
                        num_neurons[0], inputs, targets[0], 
                        learning_rate, decay_rate, beta,
                        DEFAULT, actFunc, lr_schedule, opt, lossFunc
                    )
                );
                
                // Hidden and Output Layers
                for (size_t i = 1; i < num_neurons.size(); ++i) {
                    layers.back()->feedforward();
                    
                    layers.emplace_back(make_unique<Layer>(
                        num_neurons[i], layers.back()->get_output(), targets[i],
                        learning_rate, decay_rate, beta,
                        (i == num_neurons.size() - 1 ? OUTPUT : HIDDEN),
                        actFunc, lr_schedule, opt, lossFunc
                    ));
                }

                // Initialize weight storage
                weight.resize(num_neurons.size());
                for (size_t i = 0; i < num_neurons.size(); ++i) {
                    weight[i].resize(num_neurons[i]);
                    for (size_t j = 0; j < num_neurons[i]; ++j) {
                        weight[i][j].resize(layers[i]->get_weight()[j].size());
                        for (size_t k = 0; k < layers[i]->get_weight()[j].size(); ++k) {
                            weight[i][j][k] = layers[i]->get_weight()[j][k];
                        }
                    }
                }
            }

            // Set step size for all layers
            void set_step_size(int stepsize) noexcept {
                for (auto& layer : layers) {
                    layer->set_step_size(stepsize);
                }
            }
            
            // Perform feedforward computation
            void feedforward() {
                for (auto& layer : layers) {
                    layer->feedforward();
                }
            }
            
            // Perform backpropagation
            void backpropagation() {
                for (size_t i = layers.size() - 1; i > 0; --i) {
                    layers[i]->backpropagation();
                    layers[i - 1]->set_error(layers[i]->get_error());
                }
            }
            
            // Print the state of the neural network
            void print() const {
                for (const auto& layer : layers) {
                    layer->measurement();
                }
            }
            
            // Get a reference to the layers
            const vector<unique_ptr<Layer>>& get_layers() const noexcept {
                return layers;
            }
};

#endif