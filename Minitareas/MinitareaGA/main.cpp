#include "wordGuesser.cpp"

using namespace std;

int main() {
	char answer[] = "hello";
	Guesser guesser = Guesser(answer, 5, 100, 1000, 0.01);
	
	guesser.evolve();

	/*
	//tests
	char * w = randomString(4);

	cout << "fitness test" << endl;
	cout << guesser.fitness("hello") << endl;

	cout << "mutation test" << endl;
	cout << w << endl;
	guesser.mutate(w);
	cout << w << endl;

	cout << "mating test" << endl;

	cout << guesser.mate_mutate("helpp", "dudes") << endl;
	*/
	return 0;
}