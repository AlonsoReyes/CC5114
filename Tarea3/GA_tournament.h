#include <vector>
#include <stdint.h>
#include <cmath>
#include <random>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <utility>
#include <string.h>
#include <limits>
#include "NeuralNetwork.cpp"

using namespace std;


void normalize_seeds(TrainingSet *ts) {
	vector<double> max(ts->inputs[0].size(), 0.0), min(ts->inputs[0].size(), numeric_limits<double>::max());
	for(int i = 0; i < ts->inputs.size(); i++) {
		for(int j = 0; j < ts->inputs[i].size(); j++) {
			if(ts->inputs[i][j] > max[j])
				max[j] = ts->inputs[i][j];
			if(ts->inputs[i][j] < min[j])
				min[j] = ts->inputs[i][j];
		}
	}
	for(int i = 0; i < ts->inputs.size(); i++) {
		for(int j = 0; j < ts->inputs[i].size(); j++) {
			ts->inputs[i][j] = (ts->inputs[i][j]-min[j])/(max[j] - min[j]);
		}
	}
}

TrainingSet read_seeds(string filename, int lines) {
	int n_rows = lines;
	int n_cols = 8;
	vector< vector<double> > inputs;
	vector< vector<double> > expected;
	vector< vector<string> > rows;
	ifstream file (filename);
	char const row_delim = '\n';
	char const field_delim = '\t';
	double num = 0.0;
	vector<double> one = {1.0, 0.0, 0.0};
	vector<double> two = {0.0, 1.0, 0.0};
	vector<double> three = {0.0, 0.0, 1.0};
	if(file.is_open()) {
		for(int r = 0; r < n_rows; r++) {
			vector<double> inp;
			vector<double> exp;
			for(int c = 0; c < n_cols - 1; c++) {
				file >> num;
				inp.push_back(num);
			}
			file >> num;
			if(num == 1)
				exp = one;
			else if(num == 2)
				exp = two;
			else
				exp = three;
			inputs.push_back(inp);
			expected.push_back(exp);
		}
	}
	TrainingSet ts = {inputs, expected};
	normalize_seeds(&ts);
	return ts;
}


vector<Network> init_population(int pop_size, vector<int> topology) {
	vector<Network> population;
	Network net;
	for(int i = 0; i < pop_size; i++) {
		net = Network(topology[0]);
		for(int j = 1; j < topology.size(); j++){
			net.addLayer(topology[j]);
		}
		population.push_back(net);
	}
	return population;
}


class GA {

	private:
		double fitness_goal;
		double mutation_rate;
		double best_fitness;
		double best_error;
		double gen_avg;
		int population_size;
		int generation;
		int max_iterations;
		TrainingSet test_data;
		Network best_individual;
		vector<Network> population;

	public:
		GA(double goal, double rate, int pop_size, int iterations, 
		 vector<int> topology, TrainingSet data);
		double fitness(Network* net);
		void mutate(Network* net);
		Network evolve();
		Network cross_over(Network first_parent, Network second_parent);
		Network tournament_selection(vector<Network> population, int k);
		void selection(vector<Network> population);
};

