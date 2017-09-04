from Network.Network import *
from random import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


first_net = Network(input_size=2, rate=0.5)
first_net.create_layer(3)
first_net.create_layer(2)

wb = first_net.get_network()  # Weights and bias of initial network so every net starts the same

trainList = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
resultTrain = [[1, 0], [0, 1], [0, 1], [1, 0]]

eval = 100  # number of tests
iters = 1000  # range of epochs it will be trained on

# this is used for the error curve in the figure
err_arr, epoch_arr = first_net.train(trainList, resultTrain, reps=iters, rate=0.5)
n_iter = []
acc_list = []

for k in range(iters):
    rights = 0
    net = Network()
    net.set_network(wb)
    net.train(trainList, resultTrain, reps=k, rate=0.5)
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
