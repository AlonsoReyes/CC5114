
class NPerceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def eval(self, inputs):

        if len(self.weights) != len(inputs):
            raise Exception('Need same number of inputs and weights')

        result = sum(i[0]*i[1] for i in zip(self.weights, inputs))
        return 1 if result + self.bias > 0 else 0


class Perceptron(NPerceptron):

    def eval(self, input1, input2):
        inp = [input1, input2]
        return super(Perceptron, self).eval(inp)


class NANDPerceptron(Perceptron):

    def __init__(self):
        self.weights = [-2, -2]
        self.bias = 3


class ANDPerceptron(Perceptron):

    def __init__(self):
        self.weights = [2, 2]
        self.bias = -3


class ORPerceptron(Perceptron):

    def __init__(self):
        self.weights = [2, 2]
        self.bias = -1


class BitSum():

    def __init__(self):
        self.perceptron = NANDPerceptron()

    def sum(self, input1, input2):
        p = self.perceptron
        firstLay = p.eval(input1, input2)
        secondLayOne = p.eval(firstLay, input1)
        secondLayTwo = p.eval(firstLay, input2)
        result = p.eval(secondLayOne, secondLayTwo)
        carry = p.eval(firstLay, firstLay)
        return result, carry
