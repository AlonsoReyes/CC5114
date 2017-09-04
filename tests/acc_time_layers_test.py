import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Network.Network import *
from tests.seeds_util import *

test_data, train_data, test_expected, train_expected = get_prepared_split()


eval = len(test_data)  # number of tests
iters = 200  # range of epochs it will be trained on
rate = 2.0

neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


test_networks = []
for l in layers:
    nets = []
    ss = [(l, a) for a in neurons]
    for f, s in ss:
        cur_net = Network(7)
        for i in range(f):
            cur_net.create_layer(s)
        cur_net.create_layer(3)
        nets.append(cur_net)
    test_networks.append(nets)

# number of hidden layers, number of neurons in layer, learning rate, accuracy in 100 guesses, time of training

total = []


for num_layers in range(len(test_networks)):
    lay = test_networks[num_layers]
    layer_data = []
    print("Layer:", layers[num_layers])
    for num_neurons in range(len(lay)):
        wb = lay[num_neurons].get_network()
        net = Network()
        net.set_network(wb)
        start = dt.datetime.now()
        net.train(train_data, train_expected, reps=iters, rate=rate)
        end = dt.datetime.now()
        elapsed_seconds = end - start
        rights = 0
        for i in range(eval):
            r = net.predict(test_data[i])
            d = test_expected[i]
            if r == [d.index(max(d))]:
                rights += 1

        acc = rights / eval
        test = [layers[num_layers], neurons[num_neurons], rate, acc, elapsed_seconds.total_seconds()]
        layer_data.append(test)
        print("Neuron: ", neurons[num_neurons])
    total.append(layer_data)

"""
""
with open('time_test_seed.pickle', 'wb') as handle:
    pickle.dump(total, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('time_test_seed.pickle', 'rb') as handle:
    total = pickle.load(handle)
"""
neuron_array = []
layer_array = []
time_array = []
acc_array = []
rate_array = []


for layer in total:
    for sub in layer:
        layer_array.append(sub[0])
        neuron_array.append(sub[1])
        rate_array.append(sub[2])
        acc_array.append(round(sub[3], 3))
        time_array.append(round(sub[4], 3))


time_data = {'Neurons per layer': neuron_array, 'Hidden Layers': layer_array, 'Time': time_array}
acc_data = {'Neurons per layer': neuron_array, 'Accuracy': acc_array, 'Hidden Layers': layer_array}
df = pd.DataFrame(time_data)
result = df.pivot(index='Neurons per layer', columns='Hidden Layers', values='Time')
ej = sns.heatmap(result, annot=True, fmt='g', cmap='jet', cbar_kws={'label': "Time [s]"})
ej.invert_yaxis()
plt.title('Hidden Layers vs Neurons: time needed to train Seeds dataset with 200 epochs')
plt.show()

df2 = pd.DataFrame(acc_data)
result2 = df2.pivot(index='Neurons per layer', columns='Hidden Layers', values='Accuracy')
ej2 = sns.heatmap(result2, annot=True, fmt='g', cmap='jet', cbar_kws={'label': "Accuracy"})
ej2.invert_yaxis()
plt.title('Hidden Layers vs Neurons: Effect on accuracy with rate=2.0')
plt.show()
