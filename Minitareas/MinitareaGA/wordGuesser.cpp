#include "wordGuesser.h"

using namespace std;

struct pair_comp
{
    bool operator()(const pair<char*, double> &a, const pair<char*, double>  &b) { return (a.second > b.second); }
};

Guesser::Guesser (char* t_answer, int word_length, int pop_size, int t_iterations, double rate) {
	this->answer = t_answer;
	this->answer_length = word_length;
	this->population_size = pop_size;
	this->mutation_rate = rate;
	this->max_iterations = t_iterations;
	this->generation = 0;
	this->population = Guesser::init_population(word_length, pop_size);
	this->best_guess = population[0];
}

double Guesser::fitness(char* individual) {
	double fit = 0;
	for(int i = 0; i < this->answer_length; i++) {
		if(individual[i] == this->answer[i]){
			fit += 1.0;
		}
	}
	return fit;
}

void Guesser::mutate(char* individual) {
	// Used to create random number between 0 and 1	
	random_device rd;
	mt19937 e2(rd());
	uniform_real_distribution<double> dist(0, 1);

	for(int i = 0; i < strlen(individual); i++) {
		if(dist(e2) <= this->mutation_rate){
			individual[i] = genRandom();
		}
	}

}

char* Guesser::mate_mutate(char* first_parent, char* second_parent) {
	random_device rd;
	mt19937 rng(rd());
	uniform_int_distribution<int> mate_distribution(0, this->answer_length - 1);

	int mixing_point = mate_distribution(rng);

	char* new_individual = (char*)malloc(sizeof(char)*(this->answer_length + 1));

	for(int i = 0; i < mixing_point; i++)
		new_individual[i] = first_parent[i];

	for(int i = mixing_point; i < this->answer_length; i++)
		new_individual[i] = second_parent[i];

	new_individual[this->answer_length] = '\0';

	this->mutate(new_individual);
	return new_individual;
}


int Guesser::choose_random(vector< pair <char*, double> > individuals) {
	random_device rd;
	mt19937 e2(rd());
	uniform_real_distribution<double> dist(0, 1);
	double R = dist(e2);
	if( R < individuals[0].second)
		return 0;
	for(int i = 1; i < individuals.size() - 1; i++) {
		if(individuals[i].second > R)
			return i - 1;
	}
	return individuals.size() - 1;
}


vector<char*> Guesser::select_parents(vector< pair<char*, double > > pop){
	int i = 0;
	vector<char*> parents;
	while(i < this->population_size) {
		int ind1 = this->choose_random(pop);
		int ind2 = this->choose_random(pop);
		i++;
		parents.push_back(pop[ind1].first);
		parents.push_back(pop[ind2].first);
	}
	return parents;
}


vector<char*> Guesser::selection(vector<char*> population) {
	vector< pair<char*, double> > individual_fit;
	double total_fit = 0.0;
	double current_fit = 0.0;

	for(int i = 0; i < this->population_size; i++) {
		current_fit = this->fitness(population[i]);
		if(current_fit > this->fitness(best_guess)) {
			this->best_guess = population[i];
		}

		individual_fit.push_back(make_pair(population[i], current_fit));
		total_fit += current_fit;
	}


	for(int i = 0; i < population.size(); i++) {
		individual_fit[i].second = individual_fit[i].second / total_fit;
	}

	sort(individual_fit.begin(), individual_fit.end(), pair_comp());
	//accumulation and normalization
	for(int i = 1; i < individual_fit.size(); i++) {
		individual_fit[i].second += individual_fit[i-1].second;
	}

	for(int i = 0; i < individual_fit.size(); i++)
		cout << individual_fit[i].second << endl;

	

	if(!(individual_fit[individual_fit.size() - 1].second <= 1.0 && 
		individual_fit[individual_fit.size() - 1].second > 1.0 - 0.01)) {
		cout << "Normalization done incorrectly" << endl;
	}

	vector<char*> selected_indiv = this->select_parents(individual_fit);

	return selected_indiv;
}

vector<char*> Guesser::reproduct_replace(vector<char*> selected_indiv) {
	int i = 0;
	vector<char*> new_pop;

	while(i < selected_indiv.size()) {
		char* first_parent = selected_indiv[i % this->population_size];
		char* second_parent = selected_indiv[(i + 1) % this->population_size];
		i+=2;
		new_pop.push_back(this->mate_mutate(first_parent, second_parent));
	}
	return new_pop;
}

void Guesser::evolve() {
	vector<char*> parents;
	while((this->fitness(this->best_guess) != this->answer_length) && (this->generation < this->max_iterations)) {
		cout << "Generation: " << this->generation << endl;
		parents = this->selection(this->population);
		this->population = this->reproduct_replace(parents);
		this->generation++;
		cout << "Best guess: " << this->best_guess << "--" << "Fitness: " << this->fitness(this->best_guess) << endl;
	}
}




