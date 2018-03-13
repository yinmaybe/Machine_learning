# encoding: utf-8
"""
朴素贝叶斯分类   去掉西瓜数据集中的连续属性
关于贝叶斯网(结构的选择) 有些复杂

EM算法的基本思路 具体编程需要根据问题确定E步 和M步的优化目标
"""
from sklearn import naive_bayes
import pandas as pd
import numpy as np

df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
X = df.values[:,1:7]
m, n = X.shape
attr = []
for j in range(n):
    for i in range(m):
        if X[i, j] not in attr:
            attr.append(X[i, j])
X0 = np.zeros((m, len(attr)))

# 需要改变数据集的表现方式
for i in range(m):
    for j in range(n):
        for k in range(len(attr)):
            if X[i, j] == attr[k]:
                X0[i, k] = 1

            #程序出错  因为原来变为1的属性在j的下层循环中又变为0
            #else:
                #X0[i, k] = 0


print(attr)

y = df.values[:, -1]
for i in range(len(y)):
    if y[i] == "是":
        y[i] = 1
    else:
        y[i] = -1
y = np.asarray(y, dtype = "int")
clf = naive_bayes.MultinomialNB()
clf.fit(X0, y)

sample = ["青绿", "蜷缩", "浊响", "清晰", "凹陷","硬滑"]
sample_std = np.zeros((1, len(attr)))
for i in range(len(sample)):
    for j in range(len(attr)):
        if sample[i] == attr[j]:
            sample_std[0, j ] = 1

print(sample_std, "\n\n")
print("样本预测是(1表示好瓜)(-1表示坏瓜):", clf.predict(sample_std))



