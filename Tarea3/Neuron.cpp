#include <vector>
#include <stdint.h>
#include <cmath>
#include <random>
#include <cstring>
#include <thread>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"

double Neuron::transferFunction(double x) {
	return 1.0 / (1.0 + exp(-x));
}

Neuron::Neuron(vector<double> weights, double bias) {
	n_Weights = weights;
	n_Bias = bias;
	n_Delta = 0.0;
	n_Output = 0.0;
}

double Neuron::getOutput() {
	return n_Output;
}

double Neuron::getDelta() {
	return n_Delta;
}

void Neuron::setOutput(double x) {
	n_Output = x;
}

void Neuron::setDelta(double x) {
	n_Delta = x;
}

vector<double> Neuron::getNeuron() {
	vector<double> neu = vector<double>();
	for (int i = 0; i < n_Weights.size(); i++) {
		neu.push_back(n_Weights[i]);
	}
	neu.push_back(n_Bias);
	return neu;
}

void Neuron::updateNeuron(vector<double> inputs, double learningRate) {
	assert(n_Weights.size() == inputs.size());

	for(int i = 0; i < n_Weights.size(); i++) {
		n_Weights[i] += learningRate * n_Delta * inputs[i];
	}
	n_Bias += learningRate * n_Delta;
}

double Neuron::activationFunction(vector<double> inputs) {
	assert(inputs.size() == n_Weights.size());

	double sum = 0.0;
	for(int w = 0; w < n_Weights.size(); w++) {
		sum += n_Weights[w] * inputs[w];
	}
	sum += n_Bias;
	return Neuron::transferFunction(sum);
}

void Neuron::changeNeuronWeight(int w_ind, double val) {
	this->n_Weights[w_ind] = val;
}

vector<double> Neuron::getWeights() {
	return this->n_Weights;
}

Layer::Layer(int inputSize, int neurons) {

	vector<Neuron> layerNeurons = vector<Neuron>();
	for(int i = 0; i < neurons; i++) {
		vector<double> w;
		for(int j = 0; j < inputSize; j++) {
			
			double weight = random_double(-5.0, 5.0);
			w.push_back(weight);
		} 
		double bias = random_double(-5.0, 5.0);
		layerNeurons.push_back(Neuron(w, bias));
	}
	l_neurons = layerNeurons;
	l_inputSize = inputSize;
}