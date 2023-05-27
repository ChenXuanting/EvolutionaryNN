m3 = '-2.29044175(/0.03372067(/a)(+a))(S)'
m4 = 'f(/-0.07121823)(/b(R)(+g))(+-1.30766254(+0.23611516))(+c)(S)'
m5 = '1.05588167(*k)(/0.21587891(+b))(+-0.51694618)(/-0.50769838)(S)'
m6 = '-0.15525971(+b)(+b)(*-0.40510551)(/g(/b))(/k)(S)'
m7 = '2.46027132(*k)(+e(*-1.71201650(/-1.26789814)(*2.74142959(+i)(/-0.08007508(/h))(/-0.57108328(+-0.04808928))(*2.09814987(*b))(*0.53491796(+-2.61767458)))))(+i)(+-1.74431657(+-0.56557012))(S)'
m8 = 'k(+f)(+-1.82682471)(/0.26178213)(+1.43949461)(S)'
m9 = '0.43182767(/k)(/i(*a))(*-1.46162216)(S)'

import FE11 as fe
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

x_train = x_train.T.tolist()
x_test = x_test.T.tolist()
y_train = y_train.T.tolist()
y_test = y_test.T.tolist()

f3 = fe.Compile(m3)
f4 = fe.Compile(m4)
f5 = fe.Compile(m5)
f6 = fe.Compile(m6)
f7 = fe.Compile(m7)
f8 = fe.Compile(m8)
f9 = fe.Compile(m9)

#x_test = x.T.tolist()
#y_test = y.T.tolist()

pred_3 = list(map(f3,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_4 = list(map(f4,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_5 = list(map(f5,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_6 = list(map(f6,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_7 = list(map(f7,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_8 = list(map(f8,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))
pred_9 = list(map(f9,x_test[0],x_test[1],x_test[2],x_test[3],x_test[4],x_test[5],x_test[6],x_test[7],x_test[8],x_test[9],x_test[10]))


correct = 0
preds = []
for i in range(len(pred_3)):
    poss = np.array([pred_3[i],pred_4[i],pred_5[i],pred_6[i],pred_7[i],pred_8[i],pred_9[i]])
    poss = poss/np.sum(poss)
    preds.append(poss.tolist())
    if np.argmax(poss)+3.0 == y_test[i]:
        correct += 1
    
print(correct/len(y_test))

from sklearn import metrics
from sklearn.metrics import auc
preds = np.array(preds)
y_test = np.array(y_test)
macro_roc_auc_ovr = metrics.roc_auc_score(y_test, preds, multi_class="ovr",average="macro")
print(macro_roc_auc_ovr)
