#include "AST.h"


// PARENT CLASS NODE
Node::Node(){ 
	return; 
}

Node::Node(Var value, Node* left, Node* right) {
	this->type = UNDEFINED;
	this->value = value;
	this->left = left;
	this->right = right;
}

int Node::eval() { 
	return 1; 
}

string Node::printHelper() { 
	return "";
}

void Node::print() { 
	cout << printHelper() << endl;
}

Node* Node::copy() { 
	return new Node(this->value, this->left, this->right); 
}

Node* Node::getRight() { 
	return this->right; 
}

Node* Node::getLeft() { 
	return this->left; 
}

void Node::setLeft(Node* node) {
	this->left = node;
}

void Node::setRight(Node* node) {
	this->right = node;
}


int Node::getType() { 
	return this->type; 
}

Var Node::getValue() { 
	return this->value; 
}

void Node::changeRight(Node* rr) { 
	this->right = rr->copy(); 
}

void Node::changeLeft(Node* ll) { 
	this->left = ll->copy(); 
}

void Node::toVectorHelper(vector<Node*> &vec) {
	//Node* node = new Node(this->getValue(), this->getLeft(), this->getRight());
	Node* n = this;
	this->getLeft()->toVectorHelper(vec);
	vec.push_back(n);
	this->getRight()->toVectorHelper(vec);
}

vector<Node*> Node::toVector() {
	vector<Node*> vec;
	toVectorHelper(vec);
	return vec;
}

Node* Node::selectRandomNode() {
	vector<Node*> nVec = this->toVector();
	int nInd = random_int(0, nVec.size() - 1);
	return nVec[nInd]->copy();
}


void Node::changeRandomNode(Node *node) {
	int rand = random_int(0, 1);
	int rand2 = random_int(0, 1);

	// whoops!
	if(rand == 0) {
		if(rand2 == 0 || this->getLeft()->getType() == TERMINAL) {
			this->setLeft(node);
		} else {
			this->getLeft()->changeRandomNode(node);
		}
	} else if(rand == 1) {
		if(rand2 == 0 || this->getRight()->getType() == TERMINAL) {
			this->setRight(node);
		} else {
			this->getRight()->changeRandomNode(node);
		}
	}
}




//TERMINAL NODE

TerminalNode::TerminalNode(int val) {
	this->type = TERMINAL;
	this->value.v = val;
	this->left = NULL;
	this->right = NULL;
}

int TerminalNode::eval() {
	return this->value.v; 
}

string TerminalNode::printHelper() {
	return to_string(this->getValue().v);
}

TerminalNode* TerminalNode::copy() {
	return new TerminalNode(this->getValue().v); 
}

void TerminalNode::toVectorHelper(vector<Node*> &vec) {
	Node* node = this;
	vec.push_back(node);
}




//INTERNAL NODE

InternalNode::InternalNode(Function f, Node* l, Node* r) {
	this->type = INTERNAL;
	this->value.f = f;
	this->left = l;
	this->right = r;
}

int InternalNode::eval() { 
	return this->value.f(this->getLeft()->eval(), this->getRight()->eval()); 
}

string InternalNode::printHelper() { 
	string op;
	Function f = this->value.f;
	if (f == &add) {
		op = "+";
	} else if(f == &sub) {
		op = "-";
	} else if(f == &mult) {
		op = "*";
	} else if(f == &division) {
		op = "/";
	} else {
		op = "";
	}
	return "(" + this->getLeft()->printHelper() + op + this->getRight()->printHelper() + ")"; 
}

InternalNode* InternalNode::copy() { 
	return new InternalNode(this->getValue().f, this->getLeft()->copy(), this->getRight()->copy()); 
}



AST::AST(FunctionSet f, TerminalSet t, int l) : functions(f), terminals(t), levels(l) {}	

Node* AST::createTreeHelper(int levels) {
	int randomIndex; 
	if(levels > 0) {
		randomIndex = random_int(0, this->functions.size() - 1);
		return new InternalNode(this->functions[randomIndex], createTreeHelper(levels - 1), createTreeHelper(levels - 1));
	} else {
		randomIndex = random_int(0, this->terminals.size() - 1);
		return new TerminalNode(this->terminals[randomIndex]);
	}
}


Node* AST::createTree(int levels) {
	Node * node;
	if(levels == -1) {
		node = createTreeHelper(this->levels);
	} else {
		node = createTreeHelper(levels);
	}
	return node;
}

Node* AST::createTree(int = -1);



Node* AST::crossOver(Node *lnode, Node* rnode) {
	Node *result = lnode->copy();
	Node *replacement = rnode->selectRandomNode();

	int rand = random_int(0, 1);

	if(rand == 0 || result->getType() == TERMINAL) {
		result = replacement;
	} else {
		result->changeRandomNode(replacement);
	}
	
	return result;
}