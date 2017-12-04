#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdint.h>
#include <random>
#include <cmath>
#include <climits>


using namespace std;

typedef int(*Function)(int,int);
typedef vector<Function> FunctionSet;
typedef vector<int> TerminalSet;

typedef union {
	int v;
	Function f;
} Var;


int random_int(int min, int max) {
	random_device rd;
	mt19937 rng(rd());
	uniform_int_distribution<int> int_distribution(min, max);
	
	return int_distribution(rng);
}

double random_double(int min, int max) {
	random_device rd;
	mt19937 rng(rd());
	uniform_real_distribution<double> dist(min, max);
	
	return dist(rng);
}

int add(int x, int y) {
	return x + y;
}

int sub(int x, int y) {
	return x - y;
}

int mult(int x, int y) {
	return x * y;
}

int division(int x, int y) {
	int res;
	if (y == 0) {
		//cout << "protected" << endl;
		res = 1;
	} else {
		res = x / y;
	}
	return res;
}

enum NodeType {
	UNDEFINED,
	TERMINAL,
	INTERNAL
};

class Node {

protected:
	int type;
	Var value;
	Node* left;
	Node* right;

public:
	Node();
	Node(Var value, Node* left, Node* right);
	virtual int eval();
	virtual int getType();
	virtual string printHelper();
	virtual Var getValue();
	virtual Node* copy();
	virtual Node* getLeft();
	virtual Node* getRight();
	virtual void setLeft(Node* node);
	virtual void setRight(Node* node);
	virtual Node* selectRandomNode();
	virtual void print();
	virtual void changeRight(Node* rr);
	virtual void changeLeft(Node* ll);
	virtual vector<Node*> toVector();
	virtual void toVectorHelper(vector<Node*> &vec);
	virtual void changeRandomNode(Node *node);
};

class TerminalNode : public Node {

public:
	TerminalNode(int val);

	int eval();
	string printHelper();
	TerminalNode* copy();
	void toVectorHelper(vector<Node*> &vec);

};


class InternalNode : public Node {

public:
	InternalNode(Function f, Node* l, Node* r);
	int eval();
	string printHelper();
	InternalNode* copy();
};

class AST {

private:
	FunctionSet functions;
	TerminalSet terminals;
	int levels;

public:
	AST(FunctionSet f, TerminalSet t, int l);
	Node* createTreeHelper(int levels);
	Node* createTree(int levels);
	Node* crossOver(Node* lnode, Node* rnode);
};
