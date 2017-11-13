#include <vector>
#include <stdint.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;

struct TrainingSet {
	vector< vector<double> > inputs;
	vector< vector<double> > expected;
};



int ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}	

vector<double> changeMnistLabel(double n) {
	vector<double> res;
	for(int i = 0; i < 10; i++) {
		if(n == i) res.push_back(1.0);
		else res.push_back(0.0);
	}
	return res;
}

void read_Mnist(string filename, vector<vector<double> > &vec) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i) {
            vector<double> tp;
            for(int r = 0; r < n_rows; ++r) {
                for(int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((double)temp / 255.0);
                }
            }
            vec.push_back(tp);
        }
    }
}


void read_Mnist_Label(string filename, vector< vector<double> > &vec) {
    ifstream file (filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i) {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec.push_back(changeMnistLabel((double)temp));
        }
    }
}

int random_int(int min, int max) {
	random_device rd;
	mt19937 rng(rd());
	uniform_int_distribution<int> int_distribution(min, max);
	
	return int_distribution(rng);
}

double random_double(int min, int max) {
	random_device rd;
	mt19937 rng(rd());
	uniform_real_distribution<double> dist(min, max);
	
	return dist(rng);
}


class Neuron {

	friend class Layer;

	private:
		vector<double> n_Weights;
		double n_Bias;
		double n_Output;
		double n_Delta;
		double transferFunction(double x); // Sigmoid in this implementation
		double activationFunction(vector<double> inputs);
		
		
	public:
		Neuron(vector<double> weights, double bias);
		vector<double> getNeuron();
		double getOutput();
		void setOutput(double x);
		double getDelta();
		void setDelta(double x);
		void updateNeuron(vector<double> inputs, double learningRate);
		void changeNeuronWeight(int w_ind, double val);
		vector<double> getWeights();
};


class Layer {
	friend class Network;

	private:
		vector<Neuron> l_neurons;
		int l_inputSize;
		double transferFunctionDerivative(double x);
	
	public:
		Layer(int inputSize, int neurons);
		Layer(vector < vector<double> > neuronsWeightBias);
		void updateLayer(vector<double> inputs, double learningRate);
		void backpropLastLayer(vector<double> expected);
		void backPropagate(vector<double> errors);
		void changeNeuronWeight(int n_ind, int w_ind, double val);
		void changeFullNeuron(int n_ind);
		vector< vector<double> > getLayer();
		vector<double> getErrors();
		vector<double> feedForward(vector<double> inputs);	
		vector<double> getOutputs();
		vector<Neuron> getNeurons();
};


class Network {
	friend class ParallelNet;

	inline static double errorFunction(vector<double> predicted, vector<double> expected) {
    	double sum = 0.0;
    	for(int i = 0; i < predicted.size(); i++) {
    		sum += pow((expected[i] - predicted[i]), 2.0);
    	}
    	return sum;
    }

	inline static double ClampOutputValue( double x ) {
        if ( x <= 0.5 ) return 0.0;
        else if ( x > 0.5 ) return 1.0;
        else return -1.0;
     }

	private:
		vector<Layer> n_layers;
		int n_inputSize;
		int n_outputSize;
		int total_size;
		double error;
		static double eta; //learning rate
		
	public:
		Network() = default;
		Network(int inputSize);
		Network(vector< vector< vector <double> > > net);
		void addLayer(int neurons);
		void updateNetwork(vector<double> inputs, double learningRate);
		void backPropagate(vector<double> expected);
		void setBaseNetwork(vector< vector< vector <double> > > net);
		void changeNeuronWeight(int l_ind, int n_ind, int w_ind, double val);
		void changeFullNeuron(int l_ind, int n_ind);
		double evaluateNet(TrainingSet test_data);
		vector<double> feedForward(vector<double> inputs);
		vector< vector< vector <double> > > getBaseNetwork();
		vector<double> train(vector< vector<double> > inputs, vector< vector<double> > expected, double rate, int maxEpochs);
		int getInputSize();
		int getTotalSize();
		double getError();
		vector<Layer> getLayers();
};
