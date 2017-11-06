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
#include <random>
#include <utility>

using namespace std;



// Used to get a random char from alphanum to create a random string

char genRandom() {
	static const char alphanum[] =
	"abcdefghijklmnopqrstuvwxyz";

	int stringLength = sizeof(alphanum) - 1;

	random_device rd;
	mt19937 rng(rd());
	uniform_int_distribution<int> int_distribution(0, stringLength - 1);
	int random_int = int_distribution(rng);
	char res = alphanum[random_int];
	return res;
}

char* randomString(int word_length) {

	char* word = (char*)malloc(sizeof(char)*(word_length + 1));
	for(int i = 0; i < word_length; i++) {
		word[i] = genRandom();
	}
	word[word_length] = '\0';
	//cout << word << endl;
	return word;
}


class Guesser {

	inline static vector<char*> init_population(int word_length, int pop_size) {
		vector<char*> pop;
		for(int i = 0; i < pop_size; i++) {;
			pop.push_back(randomString(word_length));
		}
		return pop;
	}

	private:
		char* answer;
		int population_size;
		int answer_length;
		int generation;
		vector<char*> population;
		char* best_guess;
		double mutation_rate;
		int max_iterations;


	public:
		Guesser(char* t_answer, int word_length, int pop_size, int t_iterations, double rate); // length of string to guess
		double fitness(char* individual);
		char* mate_mutate(char* first_parent, char* second_parent);
		void mutate(char* individual);
		vector<char*> selection(vector<char*> population);
		vector<char*> reproduct_replace(vector<char*> selected_indiv);
		void evolve();
		int choose_random(vector< pair <char*, double> > individuals);
		vector<char*> select_parents(vector< pair<char*, double > > pop);
};