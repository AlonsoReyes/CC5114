from TareaClase4.Network import *
from random import *
import matplotlib.pyplot as plt


def rand_bin_list(n):
    return [randint(0, 1) for b in range(1, n+1)]

first_net = Network(input_size=2, rate=0.5)
#first_net.create_layer(2)
first_net.create_layer(3)
first_net.create_layer(1)
wb = first_net.get_network()
#print(wb)

#testList = [list(a) for a in zip(rand_bin_list(100), rand_bin_list(100))]
testList = []
for i in range(100):
    e = int(i/25)
    if e == 0:
        testList.append([0., 0.])
    elif e == 1:
        testList.append([1., 0.])
    elif e == 2:
        testList.append([0., 1.])
    else:
        testList.append([1., 1.])

#print(testList)
shuffle(testList)
resultTest = [[int(x) ^ int(y)] for x, y in testList]

n_iter = []
acc_list = []

trainList = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
resultTrain = [[0], [1], [1], [0], [0], [1], [1], [0]]

eval = 100
iters = 1000

for k in range(iters):
    rights = 0
    net = Network()
    net.set_network(wb)
    net.train(trainList, resultTrain, k+1)
    for i in range(eval):
        a = randint(0, 1)
        b = randint(0, 1)
        out = a ^ b
        actual = net.feed_forward([a, b])
        normal = round(actual[0], 0)
        if normal == out:
            rights += 1
    n_iter.append(k)
    acc_list.append(rights/eval)

plt.plot(n_iter, acc_list, 'blue')
plt.show()


"""

[
    [
        [2.57147557, 1.89695434, 0.380250607417],
        [6.78504447, -5.92964121, 2.97653335647],
        [4.41535064, -5.97567152, -1.96438569666]
    ],
    [
        [2.80838347, -9.39005226, 9.18965719, 2.00463814965]
    ]
]

#print(first_net.get_out())
first_net.train(trainList, resultTrain, 1000)
rights = 100
#print(first_net.get_network())
#print(first_net.get_out())
for inp, exp in zip(testList, resultTest):
    result = first_net.eval(inp)
    #print(result, exp)
    print(first_net.feed_forward(inp), inp)
    if result != exp:
        rights -= 1
print(rights)

"""