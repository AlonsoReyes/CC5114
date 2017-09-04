import numpy as np


def transfer(z):
    return 1.0 / (1.0 + np.exp(-z))


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = 0
        self.delta = 0

    def activation(self, inputs):
        if len(self.weights) != len(inputs):
            raise Exception('Need same number of inputs and weights')

        s = np.dot(self.weights, inputs)+self.bias
        r = transfer(s)
        return r

    def get_neuron(self):
        weights_and_bias = []
        for w in self.weights:
            weights_and_bias.append(w)
        weights_and_bias.append(self.bias)
        return weights_and_bias

    def update_weights_bias(self, inputs, rate):
       # print(self.delta)
        for i in range(len(self.weights)):
            self.weights[i] += rate * self.delta * inputs[i]
        #print(self.delta)
        self.bias += rate * self.delta
