from TareaClase2.Perceptron import *

t = 100
xPosList = np.random.random_integers(-50, 50,  300)
yPosList = np.random.random_integers(-50, 50,  300)
perceptron = Perceptron(weights=[2, 2], bias=3, c=0.05)
trainList = [list(a) for a in zip(xPosList[:t], yPosList[:t])]
testList = [list(a) for a in zip(xPosList[t:], yPosList[t:])]

total = 0
xUpper = []
yUpper = []
xBelow = []
yBelow = []

for i in trainList:
    perceptron.train(i, curvePos(i))

for i in testList:
    if perceptron.eval(i) == 1:
        xUpper.append(i[0])
        yUpper.append(i[1])
    else:
        xBelow.append(i[0])
        yBelow.append(i[1])

    if perceptron.eval(i) == curvePos(i):
        total += 1
    else:
        total -= 1


#acc = (total/int(len(testList)))*100
#print(acc)

graph('x*3+2', range(-50, 50), xUpper, yUpper, xBelow, yBelow)
