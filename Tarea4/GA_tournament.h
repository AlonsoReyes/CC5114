#include "AST/AST.cpp"

using namespace std;


vector<Node*> init_population(int pop_size, FunctionSet fset, TerminalSet tset, int depth) {
	vector<Node*> population;
	Node* node;
	int temp_depth; 
	AST ast = AST(fset, tset, depth);
	for(int i = 0; i < pop_size; i++) {
		temp_depth = random_int(0, depth);
		node = ast.createTree(temp_depth);
		population.push_back(node);
	}
	return population;
}


class GA {

	private:
		double fitness_goal;
		double cross_over_rate;
		double mutation_rate;
		double best_fitness;
		double best_error;
		double gen_avg;
		int population_size;
		int generation;
		int max_iterations;
		int target;
		int max_depth;
		Node* best_individual;
		vector<Node*> population;
		TerminalSet terminals;
		FunctionSet functions;

	public:
		GA(double goal, double mute_rate, double cross_rate, int pop_size, int iterations, int target, int depth,
			TerminalSet tset, FunctionSet fset);
		double fitness(Node* node);
		Node* mutate(Node* node);
		Node* evolve();
		Node* cross_over(Node* first_parent, Node* second_parent);
		Node* tournament_selection(vector<Node*> population, int k);
		void selection(vector<Node*> population);
};

