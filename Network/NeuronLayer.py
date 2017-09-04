import numpy as np
from Network.Neuron import Neuron


def transfer_derivative(output):
    return output*(1.0 - output)


class NeuronLayer:

    def __init__(self, input_size=0, neurons=0):
        self.neurons = []
        self.input_size = input_size
        for i in range(neurons):
            self.neurons.append(Neuron(weights=np.random.rand(input_size), bias=np.random.rand()))

    # Neurons format [neuron1, neuron2, ...], neuronN = [weight1, weight2, ..., bias]
    def set_neurons(self, neurons, prev_size):
        w = 0
        for n in neurons:
            size = len(n)
            # Check if number of weights is equal with the number of inputs
            if prev_size is not None and (size - 1) != prev_size:
                raise Exception('Number of weights does not match with last layers number of inputs.')

            w = n[:-1]
            b = n[-1]
            new_neuron = Neuron(w, b)
            self.neurons.append(new_neuron)
        self.input_size = len(w)

    def get_layer(self):
        layer = []
        for n in self.neurons:
            layer.append(n.get_neuron())
        return layer

    def feed(self, inputs):
        layers_output = []
        for n in self.neurons:
            out = n.activation(inputs)
            n.output = out
            layers_output.append(out)
        return layers_output

    def update_last_layer(self, expected):
        for neu, exp in zip(self.neurons, expected):
            neu.delta = (exp - neu.output) * transfer_derivative(neu.output)

    def get_weights_backprop(self):
        weights = []
        for i in range(self.input_size):
            w = []
            for neuron in self.neurons:
                w.append(neuron.weights[i]*neuron.delta)
            weights.append(sum(w))
        return weights

    def get_outputs(self):
        out = []
        for neuron in self.neurons:
            if neuron.output is None:
                print("ey")
            out.append(neuron.output)
        return out

    def get_deltas(self):
        out = []
        for neuron in self.neurons:
            out.append(neuron.delta)
        return out

    def update_deltas(self, errors):
        for neu, err in zip(self.neurons, errors):
            td = transfer_derivative(neu.output)
            neu.delta = err * td

    def update(self, inputs, rate):
        for neuron in self.neurons:
            neuron.update_weights_bias(inputs, rate)
