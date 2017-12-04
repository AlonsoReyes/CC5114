#include "GA_tournament.h"


using namespace std;



//Didnt have time to make it more generic and pass functions from the classes in
//the AST file
GA::GA(double goal, double mute_rate, double cross_rate, 
	int pop_size, int iterations, int target, int depth,
	TerminalSet tset, FunctionSet fset) {
	this->generation = 0;
	this->max_iterations = iterations;
  	this->mutation_rate = mute_rate;
	this->cross_over_rate = cross_rate;
	this->fitness_goal = goal;
	this->target = target;
	this->population_size = pop_size;
	this->max_depth = depth;
	this->terminals = tset;
	this->functions = fset;
	this->population = init_population(population_size, fset, tset, depth);
}

//Turns it into a maximization problem
double GA::fitness(Node* node) {
	double result = abs(this->target - node->eval());
	if(result == 0) {
		result = INT_MAX;
	} else {
		result = 1/result;
	}
	return result;
}

//tournament selection
Node* GA::tournament_selection(vector<Node*> population, int k) {
	Node* best;
	int first = 0;
	int ind;
	double best_score;
	for(int i = 0; i < k; i++) {
		ind = random_int(0, population.size() - 1);
		if(first == 0){
			first = 1;
			best = population[ind];
			best_score = this->fitness(best);
		}
		else if(best_score < this->fitness(population[ind])) {
			best = population[ind];
		}
	}
	return best;
}


//Had to do it like this because of poor implementation of classes
Node* GA::mutate(Node* node) {
	int random_depth = random_int(0, this->max_depth);
	AST ast = AST(this->functions, this->terminals, random_depth);
	Node* mutation = ast.createTree();
	Node* result = ast.crossOver(node, mutation);
	return result;
}


Node* GA::cross_over(Node* first_parent, Node* second_parent) {
	Node* result;
	AST ast = AST(this->functions, this->terminals, this->max_depth);
	Node* f_offspring = ast.crossOver(first_parent, second_parent);
	Node* s_offspring = ast.crossOver(first_parent, second_parent);
	if(this->fitness(f_offspring) > this->fitness(s_offspring)) {
		result = f_offspring;
	} else {
		result = s_offspring;
	}

	if(random_double(0, 1) < this->mutation_rate) {
		result = this->mutate(result);
	}
	return result;
}

// selects using tournament selection the parents and then breeds
void GA::selection(vector<Node*> population) {
	int i = 0;
	vector<Node*> new_pop;
	Node *child, *first_parent, *second_parent, *best_offspring;
	double best_fit = -1;
	double fit;
	double total_fit = 0.0;
	double c_rate;


	while(i < this->population_size) {
		c_rate = random_double(0, 1);
	    first_parent = this->tournament_selection(population, 2);
	    second_parent = this->tournament_selection(population, 2);
	    if(c_rate < this->cross_over_rate) {
	    	child = this->cross_over(first_parent, second_parent);
	    	
	    } else {
	    	child = first_parent;
	    }
    	fit = this->fitness(child);
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
	this->best_error = abs(this->target - best_offspring->eval());
	this->best_fitness = best_fit;
	this->population = new_pop;
}


// Run function to write results 
Node* GA::evolve() {
	ofstream myfile;
	clock_t begin, end, t_begin, t_end;
	double time, t_time;
	time = t_time = 0.0;
	myfile.open("results/GA_tournamentResults.csv");
	myfile << "generation,best_acc,average_acc,diferencia,time\n";
	t_begin = clock();
	while(this->best_fitness < this->fitness_goal && this->generation < this->max_iterations) {
		begin = clock();
		this->selection(this->population);
		end = clock();
		time =  double(end - begin) / CLOCKS_PER_SEC;
		this->generation++;
		myfile << this->generation << "," << this->best_fitness << "," << this->gen_avg << "," << this->best_error << "," << time << '\n';

		cout << "Generation: " << this->generation << " -- Diferencia: " << this->best_error << '\n';
		 cout << "Best AST eval: " << this->best_individual->eval() << " --- Target: " << this->target << " --- AST: ";
		this->best_individual->print();
	}
	t_end = clock();
	t_time = double(t_end - t_begin) / CLOCKS_PER_SEC;
	cout << "Total time: " << t_time << '\n';
	return this->best_individual;
}
