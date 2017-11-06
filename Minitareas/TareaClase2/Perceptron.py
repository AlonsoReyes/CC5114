import matplotlib.pyplot as plt
import numpy as np


def graph(formula, x_range, dotsXU, dotsYU, dotsXB, dotsYB):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.scatter(dotsXU, dotsYU, c='blue')
    plt.scatter(dotsXB, dotsYB, c='red')
    plt.show()


def curvePos(pos, pend=3, desp=2):
    res = pos[0] * pend + desp
    return 1 if pos[1] > res else 0


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

    def __init__(self, weights, bias, c):
        self.c = c
        super(Perceptron, self).__init__(weights, bias)

    def train(self, inp, expected):
        newList = [list(a) for a in zip(self.weights, inp)]
        newWeights = []
        if expected == self.eval(inp):
            newWeights = self.weights

        elif expected > self.eval(inp):
            for i in newList:
                newWeights.append(i[0] + self.c * i[1])
        else:
            for i in newList:
                newWeights.append(i[0] - self.c * i[1])

        self.weights = newWeights


