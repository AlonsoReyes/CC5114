import random
from TareaClase4.NeuronLayer import NeuronLayer


class Network:

    # layers is an array with the number of neurons per layer ex. [2, 3 ,1] and an input_size 3
    # would mean that the neurons of the first layer receive 3 inputs each
    def __init__(self, input_size=0, rate=0.5):

        self.layers = []
        self.rate = rate
        self.input_size = input_size

    def create_layer(self, neurons_size):
        if self.input_size == 0:
            raise Exception('Cannot add layer if input size of network is not defined. '
                            'Give weights and biases to neurons with set_network'
                            'or define input size first.')

        current_size = len(self.layers)
        last_layer_size = len(self.layers[current_size - 1].neurons) if current_size != 0 else 0
        new_layer_input_size = self.input_size if current_size == 0 else last_layer_size
        self.layers.append(NeuronLayer(new_layer_input_size, neurons_size))

    # Input format [layer1, layer2, ...], layerN = [neuron1, neuron2, ...], neuronN = [weight1, weight2, ..., bias]
    def set_network(self, weights_biases):
        self.layers = []
        self.input_size = 0
        first_layer = weights_biases[0]
        self.input_size = len(first_layer[0]) - 1
        for layer in weights_biases:
            prev_size = None if len(self.layers) == 0 else len(self.layers[-1].neurons)
            lay = NeuronLayer()
            lay.set_neurons(layer, prev_size)
            self.layers.append(lay)

    def get_network(self):
        network = []
        for layer in self.layers:
            network.append(layer.get_layer())
        return network

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.feed(inputs)
        return inputs

    def back_propagation(self, expected):
        self.layers[-1].update_last_layer(expected)

        for i in reversed(range(len(self.layers[:-1]))):
            next_layer = self.layers[i + 1]
            weights = next_layer.get_weights_backprop()
            self.layers[i].update_deltas(weights)

    def update(self, inputs):
        for layer in self.layers:
            layer.update(inputs, self.rate)
            inputs = layer.get_outputs()

    def train(self, dataset, expected, reps, rate=0.5):
        self.rate = rate
        total = len(dataset)
        error_arr = []
        epoch_arr = []
        for rep in range(reps):
            rights = 0
            sum_error = 0
            for data, result in zip(dataset, expected):
                out = self.feed_forward(data)
                self.back_propagation(result)
                sum_error += sum([(result[i] - out[i])**2 for i in range(len(out))])
                self.update(data)
            error_arr.append(sum_error)
            epoch_arr.append(rep)
            for data, result in zip(dataset, expected):
                if self.predict(data) == [result.index(max(result))+1]:
                    rights += 1
            print('>epoch=%d, result=%d/%d' % (rep, rights, total))
        return error_arr, epoch_arr

    def predict(self, inputs):
        out = self.feed_forward(inputs)
        return [out.index(max(out)) + 1]

    def eval(self, inputs):
        return [1 if a >= 0.5 else 0 for a in self.feed_forward(inputs)]

    def get_info(self):
        print("Outputs")
        for layer in self.layers:
            print(layer.get_outputs())

        print("")
        print("Deltas")
        for layer in self.layers:
            print(layer.get_deltas())
        print(len(self.layers))

    def get_net_info(self):
        net = self.get_network()
        for layer in net:
            print("Layers")
            for neuron in layer:
                print(neuron)





