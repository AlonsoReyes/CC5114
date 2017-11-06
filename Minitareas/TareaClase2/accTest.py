from TareaClase2.Perceptron import *


accList = []
training = []
perceptron = Perceptron(weights=[2, 2], bias=3, c=0.01)

for k in range(200):
    t = k
    xPosList = np.random.random_integers(-50, 50, 500)
    yPosList = np.random.random_integers(-50, 50, 500)
    trainList = [list(a) for a in zip(xPosList[:t], yPosList[:t])]
    testList = [list(a) for a in zip(xPosList[t:], yPosList[t:])]
    total = 0

    for i in trainList:
        perceptron.train(i, curvePos(i))

    for i in testList:

        if perceptron.eval(i) == curvePos(i):
            total += 1
        else:
            total -= 1

    accList.append((total / int(len(testList))) * 100)
    training.append(k)

plt.plot(training, accList)
plt.show()
