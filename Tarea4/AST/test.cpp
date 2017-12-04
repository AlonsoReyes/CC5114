#include "AST.cpp"

FunctionSet fs = {&add, &sub, &mult, &division};
TerminalSet ts = {10, 1, 25, 9, 3, 6};

void basic_test() {
   TerminalNode* t = new TerminalNode(3);

   TerminalNode* t2 = new TerminalNode(2);
   InternalNode* in = new InternalNode(&add, t, t2);
   cout << t->eval() << endl;
   cout << t2->eval() << endl;
   cout << in->eval() << endl;
   in->print();

   TerminalNode* tc = t->copy();
   Node* tcc = t->copy();
   cout << t << endl;
   cout << tc << endl;
   cout << tcc << endl;
   cout << t->eval() << endl;
   cout << tc->eval() << endl;
   cout << tcc->eval() << endl;

}

void ast_test() {
   AST ast = AST(fs, ts, 2);
   Node* oldn = ast.createTree();
   Node* node = ast.createTree();

   Node* newN = oldn->copy();
   newN->changeRight(node);

   cout << "check right son, node ptr" << endl;
   cout << node << endl;
   cout << newN->getRight() << endl;
   cout << endl;

   cout << "check copy makes new ptr" << endl;
   cout << oldn << endl;
   cout << newN << endl;
   cout << endl;

   cout << "check different expr with change" << endl;
   cout << "old: ";
   oldn->print();
   cout << "node: ";
   node->print();
   cout << "changed: ";
   newN->print();
   cout << endl;

   cout << "check different values with change" << endl;
   cout << "old: ";
   cout << oldn->eval() << endl;
   cout << "node: ";
   cout << node->eval() << endl;
   cout << "changed: ";
   cout << newN->eval() << endl;
   
}

void serialize_test() {
   AST *ast = new AST(fs, ts, 2);
   Node* oldn = ast->createTree();
   Node* node = ast->createTree(2);


   cout << "BASE NODE" << endl;
   oldn->print();
   cout << oldn->eval() << endl;
   cout << endl;

   cout << "INSERT NODE" << endl;
   node->print();
   cout << node->eval() << endl;
   cout << endl;

   vector<Node*> v = oldn->toVector();

   /*
   cout << "VECTOR REP" << endl;
   for (int i = 0; i < v.size(); i++) {
      v[i]->print();
   }
   cout << endl;
   */
   cout << "NEW NODE" << endl;
   oldn = ast->crossOver(oldn, node);
   oldn->print();
   cout << oldn->eval() << endl;
}

int main() {
   //basic_test();
   //ast_test();
   serialize_test();
	return 0;
}