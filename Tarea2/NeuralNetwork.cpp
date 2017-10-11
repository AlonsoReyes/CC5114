#include <vector>
#include <stdint.h>
#include <cmath>
#include <random>
#include <cstring>
#include <thread>
#include "NeuralNetwork.h"
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>


using namespace std;


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


Layer::Layer(int inputSize, int neurons) {
	random_device rd;
	mt19937 generator(rd());
	double distributionRangeHW = 2.4 / inputSize;
	double standardDev = distributionRangeHW * 2 / 6;
	normal_distribution<> normalDistribution(0, standardDev);
	vector<Neuron> layerNeurons = vector<Neuron>();
	// Sets weights to normally distributed random values between [-2.4/inputSize, 2.4/inputSize]
	for(int i = 0; i < neurons; i++) {
		vector<double> w;
		for(int j = 0; j < inputSize; j++) {
			
			double weight = normalDistribution(generator);
			w.push_back(weight);
		} 
		double bias = normalDistribution(generator);
		layerNeurons.push_back(Neuron(w, bias));
	}
	l_neurons = layerNeurons;
	l_inputSize = inputSize;
}

Layer::Layer(vector < vector<double> > neuronsWeightBias) {
	vector<double> weights;
	double bias;
	vector<Neuron> neurons = vector<Neuron>();
	for(int neuIdx = 0; neuIdx < neuronsWeightBias.size(); neuIdx++) {
		weights = vector<double>();
		vector<double> actual = neuronsWeightBias[neuIdx];
		for(int w = 0; w < actual.size() - 1; w++) {
			weights.push_back(actual[w]);
		}
		bias = actual[actual.size() - 1];
		neurons.push_back(Neuron(weights, bias));
	}
	l_neurons = neurons;
	l_inputSize = neuronsWeightBias[0].size() - 1;
}

double Layer::transferFunctionDerivative(double x) {
	return x * (1.0 - x);
}

vector< vector<double> > Layer::getLayer() {
	vector < vector<double> > neuronsWeightBias = vector < vector<double> >();
	for(int i = 0; i < l_neurons.size(); i++) {
		neuronsWeightBias.push_back(l_neurons[i].getNeuron());
	}
	return neuronsWeightBias;
}

vector<double> Layer::getErrors() {
	vector<double> errors = vector<double>();
	for(int i = 0; i < l_inputSize; i++) {
		double sum = 0.0;
		for(int n = 0; n < l_neurons.size(); n++) {
			sum += l_neurons[n].n_Weights[i] * l_neurons[n].n_Delta;
		}
		errors.push_back(sum);
	}
	return errors;
}


void Layer::backpropLastLayer(vector<double> expected) {
	assert(l_neurons.size() == expected.size());
	for(int i = 0; i< l_neurons.size(); i++) {
		l_neurons[i].n_Delta = (expected[i] - l_neurons[i].n_Output) * Layer::transferFunctionDerivative(l_neurons[i].n_Output);
	}
}


vector<double> Layer::feedForward(vector<double> inputs) {
	vector<double> outputs = vector<double>();
	for(int n = 0; n < l_neurons.size(); n++) {
		double out = l_neurons[n].activationFunction(inputs);
		l_neurons[n].n_Output = out;
		outputs.push_back(out);
	}
	return outputs;
}


void Layer::backPropagate(vector<double> errors) {
	assert(errors.size() == l_neurons.size());
	for(int i = 0; i < l_neurons.size(); i++) {
		double td = Layer::transferFunctionDerivative(l_neurons[i].n_Output);
		l_neurons[i].n_Delta = errors[i] * td;
	}
}

void Layer::updateLayer(vector<double> inputs, double learningRate) {
	for(int n = 0; n < l_neurons.size(); n++) {
		l_neurons[n].updateNeuron(inputs, learningRate);
	}
}

vector<double> Layer::getOutputs() {
	vector<double> outputs = vector<double>();
	for(int i = 0; i < l_neurons.size(); i++) {
		outputs.push_back(l_neurons[i].getOutput());
	}
	return outputs;
}


double Network::eta = 0.15;

Network::Network(int inputSize) {
	n_layers = vector<Layer>();
	n_inputSize = inputSize;
}

Network::Network(vector< vector< vector <double> > > net) {
	n_inputSize = net[0].size();
	n_layers = vector<Layer>();
	for(int l = 0; l < net.size(); l++) {
		n_layers.push_back(Layer(net[l]));
	}
}

void Network::addLayer(int neurons) {
	int lastSize;
	if(n_layers.size() == 0) {
		lastSize = n_inputSize;
	}
	else {
		lastSize = n_layers.back().l_neurons.size();
	}
	n_layers.push_back(Layer(lastSize, neurons));
}

vector< vector< vector <double> > > Network::getBaseNetwork() {
	vector< vector< vector <double> > > net = vector< vector< vector <double> > >();
	for(int i = 0; i < n_layers.size(); i++) {
		net.push_back(n_layers[i].getLayer());
	}

	return net;
} 

void Network::setBaseNetwork(vector< vector< vector <double> > > net) {
	vector<Layer> n_layers = vector<Layer>();
	n_inputSize = net[0].size();
	n_outputSize = net[net.size() - 1].size();
	for(int i = 0; i < net.size(); i++) {
		n_layers.push_back(Layer(net[i]));
	}
}

vector<double> Network::feedForward(vector<double> inputs) {
	vector<double> inp = inputs;
	for(int i = 0; i < n_layers.size(); i++) {
		inp = n_layers[i].feedForward(inp);
	}
	return inp;
}

void Network::backPropagate(vector<double> expected) {
	n_layers.back().backpropLastLayer(expected);
	vector<double> errors;
	for(int i = n_layers.size() - 2; i >= 0; i--) {
		Layer nextLayer = n_layers[i + 1];
		errors = nextLayer.getErrors();
		n_layers[i].backPropagate(errors);
	}
}

void Network::updateNetwork(vector<double> inputs, double learningRate) {
	vector<double> inp = inputs;
	for(int i = 0; i < n_layers.size(); i++) {
		n_layers[i].updateLayer(inp, learningRate);
		inp = n_layers[i].getOutputs();
	}
}

vector<double> Network::evaluate(vector<double> inputs) {
	vector<double> results = this->feedForward(inputs);
	return results;
}

vector<double> Network::train(vector< vector<double> > inputs, vector< vector<double> > expected, double rate, int maxEpochs) {
	assert(inputs.size() == expected.size());
	int inpSize = inputs.size();
	double currentAcc = 0.0;
	int incorrectResults = 0;
	vector<double> currentCheck;
	clock_t begin, end;
	vector<double> timeVec;

	//Train
	begin = clock();
	for(int ep = 0; ep < maxEpochs; ep++) {
		for(int index = 0; index < inpSize; index++) {
			this->feedForward(inputs[index]);
			this->backPropagate(expected[index]);
			this->updateNetwork(inputs[index], rate);
		} 
		end = clock();
		timeVec.push_back(double(end - begin) / CLOCKS_PER_SEC);
	}
	return timeVec;
}

double ParallelNet::eta = 0.15;

ParallelNet::ParallelNet(int inputSize, int n_Slaves):masterNet(Network(inputSize)) {
	numberOfSlaves = n_Slaves;
	n_inputSize = inputSize;
}

void ParallelNet::addLayer(int neurons) {
	masterNet.addLayer(neurons);
}

void ParallelNet::copyMasterNet() {
	vector< vector< vector <double> > > master = masterNet.getBaseNetwork();
	for(int i = 0; i < numberOfSlaves; i++) {
		slaveNets.push_back(Network(master));
	}
}

vector< vector< vector <double> > > ParallelNet::averageSlaveNets() {
	vector< vector< vector <double> > > averagedMatrix;
	vector< vector< vector< vector <double> > > > slaveNetsMatrix;
	for(int i = 0; i < slaveNets.size(); i++) {
		slaveNetsMatrix.push_back(slaveNets[i].getBaseNetwork());
	}

	averagedMatrix = slaveNetsMatrix[0];

	for(int i = 0; i < averagedMatrix.size(); i++) {
		for(int j = 0; j < averagedMatrix[i].size(); j++) {
			for(int k = 0; k < averagedMatrix[i][j].size(); k++) {
				for(int m = 1; m < numberOfSlaves; m++) {
					averagedMatrix[i][j][k] += slaveNetsMatrix[m][i][j][k];
				}
				averagedMatrix[i][j][k] /= numberOfSlaves;
			}
		}
	}
	masterNet = Network(averagedMatrix);
	masterNet.getBaseNetwork();
	return averagedMatrix;
}


//Separates the 
vector< vector< vector < vector<double> > > > ParallelNet::dataBatches(vector< vector<double > > total_input, vector< vector<double> > expected, int numOfBatches, double sharedAmount) {
	vector< vector < vector<double> > > dataBatches;
	vector< vector < vector<double> > > expectedBatches;
	vector< vector < vector<double> > > validationBatches; // will have just 1 batch inside
	vector< vector < vector<double> > > validationExpectedBatches; // same here
	vector< vector< vector < vector<double> > > > totalData;
	int dataPerBatch;
	double sharePercentage = sharedAmount;
	vector< vector<double> > tempData;
	vector< vector<double> >tempExpected;
	vector< vector<double > > data;
	vector< vector<double > > dataExpected;
	vector< vector<double> > validationData;
	vector< vector<double> > validationDataExpected;

	// we will use the last 5% of the data as a validation set
	for(int i = 0; i < total_input.size(); i++) {
		if( i < (total_input.size() - (total_input.size() * 0.05))) {
			data.push_back(total_input[i]);
			dataExpected.push_back(expected[i]);
		} else {
			validationData.push_back(total_input[i]);
			validationDataExpected.push_back(expected[i]);
		}
	}
	validationBatches.push_back(validationData);
	validationExpectedBatches.push_back(validationDataExpected);

	cout << "Training Data Size: " << data.size() << endl;
	cout << "Validation Data Size: " << validationData.size() << endl;

	if((data.size() / numOfBatches) < 20) {
			dataPerBatch = data.size();
	} else {
		dataPerBatch = data.size() / numOfBatches;
	}

	for(int i = 0; i < numOfBatches; i++) {
		tempData = vector < vector<double> >();
		tempExpected = vector < vector<double> >();
		for(int j = 0; j < (dataPerBatch + (int)(sharePercentage * data.size())); j++) {
			tempData.push_back(data[(i*dataPerBatch+j) % data.size()]);
			tempExpected.push_back(dataExpected[(i*dataPerBatch+j) % dataExpected.size()]);
		}
		dataBatches.push_back(tempData);
		expectedBatches.push_back(tempExpected);
	}
	totalData.push_back(dataBatches);
	totalData.push_back(expectedBatches);
	totalData.push_back(validationBatches);
	totalData.push_back(validationExpectedBatches);
	return totalData;
}


void ParallelNet::train(vector< vector<double > > data, vector< vector<double> > expected, int maxEpochs_Slaves, int maxEpochs_Master, double rate, double desiredAcc) {
	eta = rate;
	thread t[numberOfSlaves];
	vector< vector< vector < vector<double> > > > separation = this->dataBatches(data, expected, numberOfSlaves+1, 0.05);
	vector< vector < vector<double> > > dataBatches = separation[0];
	vector< vector < vector<double> > > expectedBatches = separation[1];
	vector < vector<double> > validationData = separation[2][0];
	vector < vector<double> > validationExpected = separation[3][0];
	clock_t begin, end;

	int epoch = 0;
	double actualAcc = 0.0;
	double actualError;
	int incorrect;
	double time = 0.0;
	// warm up

	ofstream myfile;
    myfile.open("trainingResultsMulti4Sec.csv");
    myfile << "epoch,time,accuracy,cores\n";

	begin = clock();
	masterNet.train(data, expected, rate, 1);
	//call threads	
	while((actualAcc < desiredAcc) && (epoch < maxEpochs_Master)) {
		this->copyMasterNet();
		for(int i = 0; i < numberOfSlaves; i++) {
			t[i] = thread(trainSingleNet, &slaveNets[i], dataBatches[i], expectedBatches[i], eta, maxEpochs_Slaves);
		}
		for(int i = 0; i < numberOfSlaves; i++) {
			t[i].join();
		}
		this->averageSlaveNets();

		end = clock();
		time =  double(end - begin) / CLOCKS_PER_SEC;
		// gets the accuracy with a validation set that is not used in the training
		incorrect = 0;
		actualError = 0.0;
		for(int i = 0; i < validationData.size(); i++) {
			vector<double> res = this->evaluate(validationData[i]);
			actualError += errorFunction(res, validationExpected[i]);
			for(int j = 0; j < validationExpected[i].size(); j++) {
				if(Network::ClampOutputValue(res[j]) != validationExpected[i][j]) {
					incorrect++;
					break;
				}
			}
		}
		actualAcc = 1.0f - (incorrect/double(validationData.size()));
		epoch++;
		cout << "Actual Epoch: " << epoch  << endl;
		cout << "Actual Accuracy on Validation Data: " << actualAcc * 100.0 << endl;
		cout << "Actual Error on Vaidation Data: " << actualError << endl;
		myfile << epoch << "," << time << "," << actualAcc << "," << numberOfSlaves << "\n";
	}
	myfile.close();
	cout << "Epochs taken: " << epoch << endl;
	cout << "% Prediction on Validation data: " << actualAcc * 100.0 << endl;
}

void ParallelNet::singleTrain(vector< vector<double > > data, vector< vector<double> > expected, int maxEpochs_Master, double rate) {
	eta = rate;
	vector < vector<double> > trainData;
	vector < vector<double> > trainExpected;
	vector < vector<double> > validationData;
	vector < vector<double> > validationExpected;

	for(int i = 0; i < data.size(); i++) {
		if (i < (data.size() * 0.95)) {
			trainData.push_back(data[i]);
			trainExpected.push_back(expected[i]);
		} else {
			validationData.push_back(data[i]);
			validationExpected.push_back(data[i]);
		}
	}

	clock_t begin, end;
	int epoch = 0;
	double actualAcc = 0.0;
	double actualError;
	int incorrect;
	double time = 0.0;
	vector<double> timeVec;
	// warm up

	ofstream myfile;
    myfile.open("trainingResultsSingleSec.csv");
    myfile << "epoch,time\n";

	begin = clock();
	//call threads	
	timeVec = masterNet.train(trainData, trainExpected, rate, maxEpochs_Master);
	end = clock();
	time =  double(end - begin) / CLOCKS_PER_SEC;
	// gets the accuracy with a validation set that is not used in the training
	incorrect = 0;
	actualError = 0.0;
	for(int i = 0; i < validationData.size(); i++) {
		vector<double> res = masterNet.evaluate(validationData[i]);
		actualError += errorFunction(res, validationExpected[i]);
		for(int j = 0; j < validationExpected[i].size(); j++) {
			if(Network::ClampOutputValue(res[j]) != validationExpected[i][j]) {
				incorrect++;
				break;
			}
		}
	}
	actualAcc = 1.0f - (incorrect/double(validationData.size()));
	epoch++;
	cout << "Actual Epoch: " << epoch  << endl;
	cout << "Actual Accuracy on Validation Data: " << actualAcc * 100.0 << endl;
	cout << "Actual Error on Vaidation Data: " << actualError << endl;
	for(int i = 0; i < maxEpochs_Master; i++) {
		myfile << i << "," << timeVec[i] << "," << actualAcc << "," << 1 << "\n";
	}

	myfile.close();
	cout << "Epochs taken: " << epoch << endl;
	cout << "% Prediction on Validation data: " << actualAcc * 100.0 << endl;
}



vector<double> ParallelNet::evaluate(vector<double> inputs) {
	return masterNet.evaluate(inputs);
}


void ParallelNet::evaluateDataset(vector< vector<double > > data, vector< vector<double> > expected) {
	int incorrect = 0;
	double actualAcc;
	for(int i = 0; i < data.size(); i++) {
		vector<double> res = this->evaluate(data[i]);
		for(int j = 0; j < expected[i].size(); j++) {
			if(Network::ClampOutputValue(this->evaluate(data[i])[j]) != expected[i][j]) {
				incorrect++;
			}
		}
	}
	actualAcc = 1.0f - (incorrect/double(data.size()));
	cout << "Actual Accuracy on Test Data: " << actualAcc * 100.0 << endl;
}