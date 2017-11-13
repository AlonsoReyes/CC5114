#include "GA.h"


using namespace std;

// struct needed to sort vector
struct pair_comp
{
    bool operator()(const pair<Network, double> &a, const pair<Network, double>  &b) { return (a.second > b.second); }
};

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

// chooses the one with fitness closest to the random value 
int GA::choose_random(vector< pair <Network, double> > individuals) {
	double R = random_double(0, 1);
	if( R < individuals[0].second)
		return 0;
	for(int i = 1; i < individuals.size() - 1; i++) {
		if(individuals[i].second > R)
			return i - 1;
	}
	return individuals.size() - 1;
}

// Semi-randomly chooses parents using function choose_random
vector<Network> GA::select_parents(vector< pair<Network, double > > pop){
	int i = 0;
	vector<Network> parents;
	while(i < this->population_size) {
		int ind1 = this->choose_random(pop);
		int ind2 = this->choose_random(pop);
		parents.push_back(pop[ind1].first);
		parents.push_back(pop[ind2].first);
		i++;
	}
	return parents;
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


// choose random parents from the ones with best fitness
void GA::reproduct(vector<Network> selected_indiv) {
	int i = 0;
	vector<Network> new_pop;
	
	while(i < selected_indiv.size()) {
		Network first_parent = selected_indiv[random_int(0, selected_indiv.size() - 1)];
		Network second_parent = selected_indiv[random_int(0, selected_indiv.size() - 1)];
		
		new_pop.push_back(this->cross_over(first_parent, second_parent));
		i+=2;
	}
	this->population = new_pop;
}

// selects the mating pool
vector<Network> GA::selection(vector<Network> population) {
	vector< pair<Network, double> > individual_fit;
	double total_fit = 0.0;
	double current_fit = 0.0;
	this->best_fitness = -1;
	for(int i = 0; i < this->population_size; i++) {
		current_fit = this->fitness(&population[i]);
		if(current_fit > this->best_fitness) {
			this->best_individual = population[i];
			this->best_fitness = current_fit;
			this->best_error = this->best_individual.getError();
		}

		individual_fit.push_back(make_pair(population[i], current_fit));
		total_fit += current_fit;
	}

	this->gen_avg = total_fit / population_size;

	//descending sort
	sort(individual_fit.begin(), individual_fit.end(), pair_comp());
	
	//accumulate
	for(int i = 0; i < population.size(); i++) {
		individual_fit[i].second = individual_fit[i].second / total_fit;
	}

	//normalize
	for(int i = 1; i < individual_fit.size(); i++) {
		individual_fit[i].second += individual_fit[i-1].second;
	}

	//parent selection
	vector<Network> selected_indiv = this->select_parents(individual_fit);
	return selected_indiv;
}



// Run function, writes results too
Network GA::evolve() {
	ofstream myfile;
	clock_t begin, end, t_begin, t_end;
	double time, t_time;
	time = t_time = 0.0;
	myfile.open("test/GAResults.csv");
	myfile << "generation,best_acc,average_acc,error,time\n";
	vector<Network> new_pop;
	t_begin = clock();
	while(this->best_fitness < this->fitness_goal && this->generation < this->max_iterations) {
		begin = clock();
		this->reproduct(this->selection(this->population));
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