#include "GA_tournament.h"


using namespace std;



GA::GA(double goal, double rate, int pop_size, int iterations,
 vector<int> topology, TrainingSet data) {
  	this->mutation_rate = rate;
	this->fitness_goal = goal;
	this->population_size = pop_size;
	this->generation = 0;
	this->max_iterations = iterations;
	this->test_data = data;
	this->population = init_population(population_size, topology);
}


// result is the accuracy on the testing set
double GA::fitness(Network* net) {
	return net->evaluateNet(this->test_data);
}

//tournament selection
Network GA::tournament_selection(vector<Network> population, int k) {
	Network best;
	int first = 0;
	int ind;
	double best_score;
	for(int i = 0; i < k; i++) {
		ind = random_int(0, population.size() - 1);
		if(first == 0){
			first = 1;
			best = population[ind];
			best_score = this->fitness(&best);
		}
		else if(best_score < this->fitness(&population[ind])) {
			best = population[ind];
		}
	}
	return best;
}


// has 5 options, multiply by a factor, sum or substract one, multiply by -1
// change the weight for a random one or change the whole neuron.
void GA::mutate(Network* net) {
	int option;
	int random_layer;
	int random_neuron;
	int random_weight;
	vector<Neuron> neurons;
	vector<double> weights;
	vector<Layer> layers = net->getLayers();
	random_layer = random_int(0, layers.size() - 1);
	neurons = layers[random_layer].getNeurons();
	random_neuron = random_int(0, neurons.size() - 1);
	weights = neurons[random_neuron].getWeights();
	random_weight = random_int(0, weights.size() - 1);

	option = random_int(0, 5);

	if(option == 0) {
		net->changeNeuronWeight(random_layer, random_neuron, random_weight, weights[random_weight]*random_double(0.5, 1.5));
	} else if(option == 1) {
		net->changeNeuronWeight(random_layer, random_neuron, random_weight, weights[random_weight]+random_double(0.0, 1.0));
	} else if(option == 2) {
		net->changeNeuronWeight(random_layer, random_neuron, random_weight, weights[random_weight]-random_double(0.0, 1.0));
	} else if(option == 3) {
		net->changeNeuronWeight(random_layer, random_neuron, random_weight, weights[random_weight]*(-1.0));
	} else if(option == 4) {
		net->changeNeuronWeight(random_layer, random_neuron, random_weight, random_double(-5.0, 5.0));
	} else {
		net->changeFullNeuron(random_layer, random_neuron);
	}
}

//Create layer and neurons with one parent until a certain random neuron
//then the rest with the corresponding part of the other parent, then mutate
Network GA::cross_over(Network first_parent, Network second_parent) {
	vector< vector< vector <double> > > first_off_spring;
	vector< vector< vector <double> > > second_off_spring;
	vector< vector< vector <double> > > first_net = first_parent.getBaseNetwork();
	vector< vector< vector <double> > > second_net = second_parent.getBaseNetwork();
	int total_layers = first_net.size();
	int layer_index = random_int(0, first_net.size() - 1);
	int layer_neurons = first_net[layer_index].size();
	int neuron_index = random_int(0, layer_neurons);
	for(int l_i = 0; l_i < total_layers; l_i++) {
		if(l_i < layer_index) {
			first_off_spring.push_back(first_net[l_i]);
			second_off_spring.push_back(second_net[l_i]);
		} else if(l_i > layer_index) {
			first_off_spring.push_back(second_net[l_i]);
			second_off_spring.push_back(first_net[l_i]);
		} else {
			vector< vector<double> > first_new_layer;
			vector< vector<double> > second_new_layer;
			for(int i = 0; i < neuron_index; i++) {
				first_new_layer.push_back(first_net[layer_index][i]);
				second_new_layer.push_back(second_net[layer_index][i]);
			}
			for(int i = neuron_index; i < layer_neurons; i++) {
				first_new_layer.push_back(second_net[layer_index][i]);
				second_new_layer.push_back(first_net[layer_index][i]);
			}
			first_off_spring.push_back(first_new_layer);
			second_off_spring.push_back(second_new_layer);
		}
	}

	//pre selection of best fitness between parents and offsprings
	Network result;
	
	Network f_offspring = Network(first_off_spring);
	Network s_offspring = Network(second_off_spring);

	if(this->fitness(&f_offspring) > this->fitness(&s_offspring)) {
		result = f_offspring;
	} else {
		result = s_offspring;
	}

	if(random_double(0, 1) < this->mutation_rate) {
		this->mutate(&result);
	}
	return result;
}

// selects using tournament selection the parents and then breeds
void GA::selection(vector<Network> population) {
	int i = 0;
	vector<Network> new_pop;
	Network child;
	Network first_parent, second_parent;
	Network best_offspring;
	double best_fit = -1;
	double fit;
	double total_fit = 0.0;


	while(i < this->population_size) {
	    first_parent = this->tournament_selection(population, 2);
	    second_parent = this->tournament_selection(population, 2);
	    child = this->cross_over(first_parent, second_parent);

    	fit = this->fitness(&child);
    	total_fit += fit;
    	if(best_fit < fit){
    		best_fit = fit;
    		best_offspring = child;
    	}
    	new_pop.push_back(child);
	    
		i++;
	}
	this->gen_avg = total_fit / this->population_size;
	this->best_individual = best_offspring;
	this->best_error = best_offspring.getError();
	this->best_fitness = best_fit;
	this->population = new_pop;
}


// Run function to write results 
Network GA::evolve() {
	ofstream myfile;
	clock_t begin, end, t_begin, t_end;
	double time, t_time;
	time = t_time = 0.0;
	myfile.open("test/GA_tournamentResults.csv");
	myfile << "generation,best_acc,average_acc,error,time\n";
	t_begin = clock();
	while(this->best_fitness < this->fitness_goal && this->generation < this->max_iterations) {
		begin = clock();
		this->selection(this->population);
		end = clock();
		time =  double(end - begin) / CLOCKS_PER_SEC;
		this->generation++;
		myfile << this->generation << "," << this->best_fitness << "," << this->gen_avg << "," << this->best_error << "," << time << '\n';

		cout << "Generation: " << this->generation << " -- Best Result: " << this->best_fitness << " -- Error: " << this->best_individual.getError()
		 << " -- Gen Avg: " << this->gen_avg << '\n';
	}
	t_end = clock();
	t_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
	cout << "Total time: " << t_time << '\n';
	return this->best_individual;
}
