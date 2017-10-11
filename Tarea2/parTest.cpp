#include <vector>
#include <stdint.h>
#include <cmath>
#include <random>
#include "NeuralNetwork.cpp"


using namespace std;


int main() {

	ParallelNet pnet = ParallelNet(784, 4);
	pnet.addLayer(100);
	pnet.addLayer(10);
	vector< vector<double> > train_mnist_img;
	vector< vector<double> > train_mnist_label;
	vector< vector<double> > test_mnist_img;
	vector< vector<double> > test_mnist_label;
	pnet.read_Mnist("mnist/train-images.idx3-ubyte", train_mnist_img);
	pnet.read_Mnist_Label("mnist/train-labels.idx1-ubyte", train_mnist_label);
	pnet.read_Mnist("mnist/t10k-images.idx3-ubyte", test_mnist_img);
	pnet.read_Mnist_Label("mnist/t10k-labels.idx1-ubyte", test_mnist_label);
	pnet.train(train_mnist_img, train_mnist_label, 1, 20, 1.0, 0.97);
	pnet.evaluateDataset(test_mnist_img ,test_mnist_label);
	
	return 0;
}