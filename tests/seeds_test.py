import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from Network.Network import *
from tests.seeds_util import *


# Gets the data in a format that the network understands it. For example the output [1] is [1, 0, 0]
# This gets the data normalized too.
test_data, train_data, test_expected, train_expected = get_prepared_split()

eval = len(test_data)  # number of test data
iters = 100  # maximum number of epochs to train with
rate = 2.0  # learning rate

# base network
net = Network(7)
net.create_layer(5)
net.create_layer(3)
wb = net.get_network()  # base weights and biases for the test

n_iter = []
acc_list = []

# get the error for each epoch
err_arr, epoch_arr = net.train(train_data, train_expected, iters, rate)

# train with different amout of epochs starting from the base network each time
for k in range(iters):
    rights = 0
    net = Network()
    net.set_network(wb)
    net.train(train_data, train_expected, k, rate)
    for i in range(eval):
        r = net.predict(test_data[i])
        d = test_expected[i]
        if r == [d.index(max(d))]:
            rights += 1

    n_iter.append(k)
    acc_list.append(rights / eval)


# plots two curves with different scales in the same figure
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, epoch_arr, err_arr, acc_list, 'r', 'b', x_label='Epochs', y1_label='Error', y2_label='Accuracy')
plt.title("Seeds dataset with {} test iterations".format(eval))
#plt.title("Seeds dataset without shuffling training set")
red_patch = mpatches.Patch(color='red', label='Error')
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
print(rights*100/len(test_data))
