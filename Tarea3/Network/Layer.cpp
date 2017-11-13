#include "Neuron.cpp"

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

void Layer::changeNeuronWeight(int n_ind, int w_ind, double val) {
	this->l_neurons[n_ind].changeNeuronWeight(w_ind, val);
}

void Layer::changeFullNeuron(int n_ind) {
	int neuron_weights = this->l_neurons[n_ind].getWeights().size();
	vector<double> weights;
	double bias;
	// Sets weights to normally distributed random values between [-2.4/inputSize, 2.4/inputSize]
	for(int j = 0; j < neuron_weights; j++) {
		double weight = random_double(-5.0, 5.0);
		weights.push_back(weight);
	} 
	bias = random_double(-5.0, 5.0);
	this->l_neurons[n_ind] = Neuron(weights, bias);
}


vector<Neuron> Layer::getNeurons() {
	return this->l_neurons;
}
