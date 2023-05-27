import FE11
import numpy as np
import random
import math
import csv

dataset = []
with open('winequality-white.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        dataset.append(row)

for i in range(len(dataset)):
    dataset[i] = [float(item) for item in dataset[i][0].split(";")]

dataset = np.array(dataset)
x = dataset.T[:11].T
x = (x - np.min(x,axis=0))/(np.max(x,axis=0) - np.min(x,axis=0))
y = dataset.T[11].T
y[y!=3] = 0.0
y[y==3] = 1.0

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train = x_train.T.tolist()
x_test = x_test.T.tolist()
y_train = y_train.T.tolist()
y_test = y_test.T.tolist()


def evaluate(f, inputs, labels):
    predictions = list(map(f,inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8],inputs[9],inputs[10]))
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    if np.any(np.isnan(predictions)):
        return None,None,False
    
    predictions[predictions == 1.0] -= 0.0000000000001
    predictions[predictions == 0.0] += 0.0000000000001

    loss = np.mean(-labels*np.log(predictions)-(1-labels)*np.log(1-predictions))
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0

    accuracy = 100 - np.mean(np.abs(predictions - labels)) * 100
    return loss, accuracy, True

population = 10000

machines = [random.choice(FE11.inputs) for i in range(population)]
for i in range(5):
    for j in range(len(machines)):
        machines[j] = FE11.mutate(machines[j],0.3,False)


bestf = ''
bestperf = math.inf
bestacc = 0

for i in range(10000):
    nextGen = []
    perfs = []
    broken = []
    for j in range(len(machines)):
        bestindiv = ''
        bestpref_indiv = math.inf
        bestacc_indiv = 0
        for k in range(10):
            if k == 0:
                indiv = machines[j]
            else:
                indiv = FE11.para_learning(machines[j],0.3)
            perf, acc, valid = evaluate(FE11.Compile(indiv+'(S)'),x_train,y_train)
            if valid and perf < bestpref_indiv:
                bestpref_indiv = perf
                bestindiv = indiv
                bestacc_indiv = acc
        if bestindiv == '':
            broken.append(j)
        else:
            if bestpref_indiv < bestperf:
                bestperf = bestpref_indiv
                bestf = bestindiv
                bestacc = bestacc_indiv
            perfs.append(bestpref_indiv)
            machines[j] = bestindiv
    nperfs = np.array(perfs)
    med = np.median(nperfs)
    ng = 0
    ns = 0
    for perf in perfs:
        if perf>= med:
            ng+=1
        else:
            ns+=1
    print(ng)
    print(ns)
    c = 0
    for j in range(len(machines)):
        if j in broken:
            c += 1
        else:
            if perfs[j-c] <= med:
                nextGen.append(FE11.mutate(machines[j],0.02,True))
                if FE11.decision(0.5):
                    nextGen.append(FE11.mutate(machines[j],0.1,True))
            else:
                if FE11.decision(0.5):
                    nextGen.append(FE11.mutate(machines[j],0.02,True))
    machines = [FE11.mutate(bestf,0.01,True), FE11.mutate(bestf,0.05,True), FE11.mutate(bestf,0.1,True)]
    survive = 1/len(nextGen)*population
    for machine in nextGen:
        if FE11.decision(survive):
            machines.append(machine)
    print('Best performance in '+str(i+1)+'th generation: '+str(min(perfs)))
    print('All-time best performance: '+str(bestperf))
    print('All-time highest accuracy: '+str(bestacc))
    print('All-time best performer: '+ bestf)



