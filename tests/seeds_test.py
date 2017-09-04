import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from Network.Network import *
from tests.seeds_util import *

net = Network(7)
net.create_layer(5)
net.create_layer(3)
wb = net.get_network()

# Gets the data in a format that the network understands it. For example the output [1] is [1, 0, 0]
# This gets the data normalized too.
test_data, train_data, test_expected, train_expected = get_prepared_split()

n_iter = []
acc_list = []
net = Network()
net.set_network(wb)

eval = len(test_data)
iters = 100
rate = 2.0

err_arr, epoch_arr = net.train(train_data, train_expected, iters, rate)

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


fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, epoch_arr, err_arr, acc_list, 'r', 'b', x_label='Epochs', y1_label='Error', y2_label='Accuracy')
plt.title("Seeds dataset with {} test iterations".format(eval))
#plt.title("Seeds dataset without shuffling training set")
red_patch = mpatches.Patch(color='red', label='Error')
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
print(rights*100/len(test_data))
