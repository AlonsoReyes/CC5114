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
#include "GA.cpp"

using namespace std;


int main() {

	TrainingSet ts = read_seeds("seeds/seeds.txt", 210);
	//TrainingSet ts_2 = read_seeds("seeds_firsthalf.txt", 105);
	TrainingSet test_set = read_seeds("seeds/seeds_secondhalf.txt",105);

	vector<int> topology = {7, 5, 3};
	GA gen = GA(0.95, 0.3, 1000, 500, topology, ts);
	Network best = gen.evolve();
	cout << best.evaluateNet(test_set) << endl;

	return 0;
}