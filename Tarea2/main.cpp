#include <vector>
#include <stdint.h>
#include <cmath>
#include <random>
#include "NeuralNetwork.cpp"


using namespace std;


int main() {
	vector< vector<double> > inputs;
	vector< vector<double> > expected;
	double inp[] = {0.0, 0.0};
	vector<double> inpVal (inp, inp + sizeof(inp) / sizeof(double) );
	inputs.push_back(inpVal);
	double inp2[] = {0.0, 1.0};
	vector<double> inpVal2 (inp2, inp2 + sizeof(inp2) / sizeof(double) );
	inputs.push_back(inpVal2);
	double inp3[] = {1.0, 0.0};
	vector<double> inpVal3 (inp3, inp3 + sizeof(inp3) / sizeof(double) );
	inputs.push_back(inpVal3);
	double inp4[] = {1.0, 1.0};
	vector<double> inpVal4 (inp4, inp4 + sizeof(inp4) / sizeof(double) );
	inputs.push_back(inpVal4);


	double exptd[] = {0.0};
	vector<double> exptdVal (exptd, exptd + sizeof(exptd) / sizeof(double) );
	expected.push_back(exptdVal);
	double exptd2[] = {1.0};
	vector<double> exptdVal2 (exptd2, exptd2 + sizeof(exptd2) / sizeof(double) );
	expected.push_back(exptdVal2);
	double exptd3[] = {1.0};
	vector<double> exptdVal3 (exptd3, exptd3 + sizeof(exptd3) / sizeof(double) );
	expected.push_back(exptdVal3);
	double exptd4[] = {0.0};
	vector<double> exptdVal4 (exptd4, exptd4 + sizeof(exptd4) / sizeof(double) );
	expected.push_back(exptdVal4);


	Network net = Network(2);
	net.addLayer(3);
	net.addLayer(1);
	//net.getBaseNetwork();
	net.train(inputs, expected, 0.5, 2000);
	//net.getBaseNetwork();
	Network net2 = Network(net.getBaseNetwork());
	//net2.train(inputs, expected, 0.5, 1000);
	vector<double> rrr;
	rrr.push_back(1.0);
	rrr.push_back(1.0);
	vector<double> res = net2.evaluate(rrr);
	for(r : res) {
		cout << "Result: " << r << endl;
	}
	
	return 0;
}