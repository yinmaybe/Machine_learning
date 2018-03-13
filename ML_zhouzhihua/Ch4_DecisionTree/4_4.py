"""
sklearn库中的决策树只支持number，所以只能产生二叉树，且只能通过参数设置控制树，
没有所谓的预剪枝和后剪枝
"""
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import pydotplus
import os
import pandas as pd
df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
dataSet = df.values[:,1:]
dt = DecisionTreeClassifier()

os.environ["PATH"] += os.pathsep + r'G:\Graphviz\bin'
#DecisionTreeClassifier不支持字符串
# 将样本中的字符属性值转换为数字
m, n = dataSet.shape
for i in range(n):
    if type(dataSet[0, i]).__name__ == "float" or type(dataSet[0, i]).__name__ == "int":
        pass
    else:
        dict_key = {}
        num = 0
        for j in range(m):
            if dataSet[j, i] not in dict_key:
                dict_key[dataSet[j, i]] = num
                num += 1
            dataSet[j, i] = dict_key[dataSet[j, i]]
        print(dict_key)

X = dataSet[:,:n-1]
y = dataSet[:, n-1].ravel().tolist()
dt = dt.fit(X, y)
dot_data = export_graphviz(dt, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("watermelon.pdf")
