from TareaClase3.Sigmoid import *


accListP = []
training = []
accListS = []
perceptron = Perceptron(weights=[2, 2], bias=3)
neuron = Sigmoid(weights=[2, 2], bias=3)
data = 500
for k in range(200):
    t = k
    totalP = data - k
    totalS = data - k
    xPosList = np.random.random_integers(-50, 50, data)
    yPosList = np.random.random_integers(-50, 50, data)
    trainList = [list(a) for a in zip(xPosList[:t], yPosList[:t])]
    testList = [list(a) for a in zip(xPosList[t:], yPosList[t:])]

    for i in trainList:
        perceptron.train(i, curvePos(i), c=0.01)
        neuron.train(i, curvePos(i), c=0.01, threshold=0.5)

    for i in testList:
        if perceptron.eval(i) != curvePos(i):
            totalP -= 1

        if neuron.eval(i, threshold=0.5) != curvePos(i):
            totalS -= 1

    accListP.append((totalP / int(len(testList))) * 100)
    training.append(k)
    accListS.append((totalS / int(len(testList))) * 100)

plt.plot(training, accListP, 'blue')

plt.plot(training, accListS, 'red')
plt.show()
