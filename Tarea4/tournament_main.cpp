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
#include "GA_tournament.cpp"

using namespace std;


int main() {
	FunctionSet fs = {&add, &sub, &mult, &division};
	TerminalSet ts = {25, 7, 8, 10, 4, 2};
	int target = 400;
	int iterations = 50;
	int population = 100;
	int depth = 3;
	double goal = 1;
	double mutation_rate = 0.01;
	double cross_over_rate = 0.9;

	GA gen = GA(goal, mutation_rate, cross_over_rate, population, iterations, target, depth, ts, fs);
	Node* best = gen.evolve();
	cout << "Evolve result: ";
	best->print();

	return 0;
}