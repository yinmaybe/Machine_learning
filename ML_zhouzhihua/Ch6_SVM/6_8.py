#encoding: utf-8
"""
核函数中的各个参数有什么意义  如何调参？？
"""
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入数据
df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
dataset = df.values[:, 7:9]
X = dataset[:, 0]
X_2 = X[:, np.newaxis]

y = dataset[:,1]

clf = SVR(C=1e+6, epsilon=0.02)
clf.fit(X_2, y)

x_min, x_max = X.min()-0.3, X.max()+0.3
y_min, y_max = y.min()-0.1, y.max()-0.1
xx = np.arange(x_min, x_max, 0.002)
xx_2 = xx[:, np.newaxis]
y_predict = clf.predict(xx_2)
plt.scatter(X, y, c="b", s=20, edgecolors='k')
plt.plot(xx, y_predict)
plt.xlabel("Density")
plt.ylabel("Sugar Ratio")
plt.title("SVR with RBF")
plt.show()
print(clf.dual_coef_)
print(clf.support_vectors_)