import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from Network.Network import *
from tests.seeds_util import *


# Gets the data in a format that the network understands it. For example the output [1] is [1, 0, 0]
# This gets the data normalized too.
test_data, train_data, test_expected, train_expected = get_prepared_split()
eval = len(test_data)
iters = 100

rates = []  # rates to be trained with: 0.2 to 20.0
for i in range(100):
    rates.append((i+1)*0.2)

# base network
net = Network(7)
net.create_layer(5)
net.create_layer(3)
wb = net.get_network()  # base weights and biases for the test

acc_list = []


# train with the different rates starting from the base network each time
for k in range(len(rates)):
    rights = 0
    net = Network()
    net.set_network(wb)
    net.train(train_data, train_expected, iters, rates[k])
    for i in range(eval):
        r = net.predict(test_data[i])
        d = test_expected[i]
        if r == [d.index(max(d))]:
            rights += 1

    acc_list.append(rights / eval)


plt.plot(rates, acc_list)
plt.title("Accuracy with 100 epochs and different rates")
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[blue_patch])
plt.ylabel('Accuracy')
plt.xlabel('Learning Rates')
plt.show()
