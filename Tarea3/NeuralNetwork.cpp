#include "Layer.cpp"

using namespace std;

double Network::eta = 0.15;


Network::Network(int inputSize) {
	this->error = 1.0;
	this->n_layers = vector<Layer>();
	this->n_inputSize = inputSize;
	this->total_size = 0;
}

Network::Network(vector< vector< vector <double> > > net) {
	this->error = 1.0;
	this->n_inputSize = net[0][0].size() - 1;
	this->n_layers = vector<Layer>();
	for(int l = 0; l < net.size(); l++) {
		this->n_layers.push_back(Layer(net[l]));
		for(int n = 0; n < net[l].size(); n++) {
			this->total_size += net[l][n].size();
		}
	}
}

void Network::addLayer(int neurons) {
	int lastSize;
	if(n_layers.size() == 0) {
		lastSize = this->n_inputSize;
	}
	else {
		lastSize = this->n_layers.back().l_neurons.size();
	}
	this->n_layers.push_back(Layer(lastSize, neurons));
	this->total_size += neurons*(lastSize + 1);
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
	n_inputSize = net[0][0].size() - 1;
	n_outputSize = net[net.size() - 1].size();
	for(int i = 0; i < net.size(); i++) {
		n_layers.push_back(Layer(net[i]));
	}
}

int Network::getInputSize() {
	return this->n_inputSize;
}

int Network::getTotalSize() {
	return this->total_size;
}

void Network::changeNeuronWeight(int l_ind, int n_ind, int w_ind, double val) {
	this->n_layers[l_ind].changeNeuronWeight(n_ind, w_ind, val);
}

void Network::changeFullNeuron(int l_ind, int n_ind) {
	this->n_layers[l_ind].changeFullNeuron(n_ind);
}

vector<double> Network::feedForward(vector<double> inputs) {
	vector<double> inp = inputs;
	for(int i = 0; i < n_layers.size(); i++) {
		inp = n_layers[i].feedForward(inp);
	}
	return inp;
}

vector<Layer> Network::getLayers() {
	return this->n_layers;
}

double Network::getError() {
	return this->error;
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


double Network::evaluateNet(TrainingSet test_data) {
	vector<double> results;
	vector< vector<double> > inputs = test_data.inputs;
	vector< vector<double> > expected = test_data.expected;
	int incorrect = 0;
	double actualAcc = 0.0;
	double actualError = 0.0;
	for(int i = 0; i < inputs.size(); i++) {
		results = this->feedForward(inputs[i]);
		actualError += Network::errorFunction(results, expected[i]);
		for(int j = 0; j < expected[i].size(); j++) {
			if(Network::ClampOutputValue(results[j]) != expected[i][j]) {
				//cout << results[j] << " " << expected[i][j] << endl;
				incorrect++;
				break;
			}
		}
	}
	this->error = actualError;
	actualAcc = 1.0f - (incorrect/double(inputs.size()));
	return actualAcc;
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

