# encoding: utf-8
"""
使用logistic回归  的多变量决策树
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from collections import deque
import numpy as np
import pandas as pd
df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
dataSet = df.values[:,7:]
m, n = dataSet.shape
label_dict = {"是":1, "否":0}
for i in range(m):
    if dataSet[i,n-1] == "是":
        dataSet[i, n-1] = 1
    else:
        dataSet[i, n-1] = 0

X = dataSet[:,:n-1]
y = dataSet[:, n-1].tolist()
lgr = linear_model.LogisticRegression(C=1e5)
weight_list = []
dq_X = []
dq_y = []
dq_X.append(X)
dq_y.append(y)
while len(dq_X)!=0:
    X, y = np.array(dq_X.pop()),np.array(dq_y.pop())
    m, n = X.shape
    # 判断是否还要再划分
    lgr.fit(X, y)
    pre_label = lgr.predict(X)
    if all(pre_label== y):
        W = None
        weight_list.append(W)
        continue
    W = lgr.coef_
    b = lgr.intercept_

    # W的值时None  ？？ 猜测：array和list间混乱了
    #W = W.tolist().append(b)
    cof = []
    cof.append(W)
    cof.append(b)
    weight_list.append(cof)
    X_1 = []
    X_2 = []
    y_1 = []
    y_2 = []
    for i in range(m):
        if pre_label[i] == 1:
            X_1.append(X[i])
            y_1.append(y[i])
        else:
            X_2.append(X[i])
            y_2.append(y[i])
    dq_X.append(X_1)
    dq_X.append(X_2)
    dq_y.append(y_1)
    dq_y.append(y_2)

print(weight_list)