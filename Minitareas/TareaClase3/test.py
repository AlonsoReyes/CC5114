from TareaClase3.Sigmoid import *

t = 100
xPosList = np.random.random_integers(-50, 50,  300)
yPosList = np.random.random_integers(-50, 50,  300)
neuron = Sigmoid(weights=[2, 2], bias=3)
trainList = [list(a) for a in zip(xPosList[:t], yPosList[:t])]
testList = [list(a) for a in zip(xPosList[t:], yPosList[t:])]
c = 0.01
total = 0
xUpper = []
yUpper = []
xBelow = []
yBelow = []
threshold = 0.5
for i in trainList:
    neuron.train(i, curvePos(i), c, threshold)

for i in testList:
    ev = neuron.eval(i, threshold)
    if ev == 1:
        xUpper.append(i[0])
        yUpper.append(i[1])
    else:
        xBelow.append(i[0])
        yBelow.append(i[1])

    if ev == curvePos(i):
        total += 1
    else:
        total -= 1


acc = (total/int(len(testList)))*100
print(acc)
graph('x*3+2', range(-50, 50), xUpper, yUpper, xBelow, yBelow)

