from Network.Network import *
from random import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


trainList = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
resultTrain = [[1, 0], [0, 1], [0, 1], [1, 0]]

eval = 100  # number of tests
iters = 1000  # range of epochs it will be trained on
rate = 0.5  # learning rate


# base network
first_net = Network(input_size=2)
first_net.create_layer(3)
first_net.create_layer(2)

wb = first_net.get_network()  # base weights and biases for the test

# this is used for the error curve in the figure
err_arr, epoch_arr = first_net.train(trainList, resultTrain, reps=iters, rate=rate)
n_iter = []
acc_list = []


# train with different epochs and get the accuracy
for k in range(iters):
    rights = 0
    net = Network()
    net.set_network(wb)
    net.train(trainList, resultTrain, reps=k, rate=rate)
    for i in range(eval):
        a = randint(0, 1)
        b = randint(0, 1)
        out = [a ^ b]
        prediction = net.predict([a, b])
        if prediction == out:
            rights += 1
    n_iter.append(k)
    acc_list.append(rights/eval)


fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, epoch_arr, err_arr, acc_list, 'r', 'b', x_label='Epochs', y1_label='Error', y2_label='Accuracy')
plt.title("XOR prediction with {} test iterations".format(eval))
red_patch = mpatches.Patch(color='red', label='Error')
blue_patch = mpatches.Patch(color='blue', label='Accuracy')
plt.legend(handles=[red_patch, blue_patch])
plt.show()
