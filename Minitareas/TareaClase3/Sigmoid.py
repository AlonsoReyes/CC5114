import numpy as np
import matplotlib.pyplot as plt


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


class Sigmoid:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def eval(self, inputs, threshold):

        if len(self.weights) != len(inputs):
            raise Exception('Initialized weights with more than 2 values')

        result = sum(i[0]*i[1] for i in zip(self.weights, inputs)) + self.bias
        div = 1 + np.exp(-result)
        result = 1/div
        return 1 if result >= threshold else 0

    def train(self, inp, expected, c, threshold):
        newList = [list(a) for a in zip(self.weights, inp)]
        newWeights = []
        e = self.eval(inp, threshold)
        if expected == e:
            newWeights = self.weights

        elif expected > e:
            for i in newList:
                newWeights.append(i[0] + c * i[1])
        else:
            for i in newList:
                newWeights.append(i[0] - c * i[1])

        self.weights = newWeights


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

    def train(self, inp, expected, c):
        newList = [list(a) for a in zip(self.weights, inp)]
        newWeights = []
        if expected == self.eval(inp):
            newWeights = self.weights

        elif expected > self.eval(inp):
            for i in newList:
                newWeights.append(i[0] + c * i[1])
        else:
            for i in newList:
                newWeights.append(i[0] - c * i[1])

        self.weights = newWeights