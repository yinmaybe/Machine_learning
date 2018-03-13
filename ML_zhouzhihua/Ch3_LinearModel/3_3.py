# encoding: utf8
"""
编程实现对率回归，并给出西瓜数据集3.0a 上的结果
"""

from sklearn import linear_model
import numpy as np
from  matplotlib import pyplot as plt
from  matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simkai.ttf", size=17 )

file1 = open(r'../watermelon.csv','r',encoding="utf-8")
data = [line.strip('\n').split(',') for line in file1]
X = [[float(raw[-3]), float(raw[-2])] for raw in data[1:]]
# 很奇怪的是 print(Y[0] == "是") 输出结果是false ？？
# 懂了  其实Y[0]的实际值是"是  "   注意有两个空格
# utf-8是Unicode是一种实现方式
Y = [raw[-1].strip() for raw in data[1:]]
Y = [1 if y == "是" else 0 for y in Y]
X = np.array(X)
Y = np.array(Y)
file1.close()
# import some data to play with
h = .002  # step size in the mesh

# C 是衡量正则化项和l2 惩罚性的相对权值
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# np._c  np._r  column row  按列或按行重组
test = np.c_[xx.ravel(), yy.ravel()]
#print test
Z = logreg.predict(test)
#print Z
#np.c_() 连接多个数组为一个数组
# Put the result into a color plot
Z = Z.reshape(xx.shape)
#print Z
#plt.figure(1, figsize=(4, 3))
# 可以进行复杂颜色设置的plot
#Plot a quadrilateral mesh.
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
# edgecolors 是marker的边线
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('密度',fontproperties = font )
plt.ylabel('含糖率', fontproperties = font)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
# 设置横轴和纵轴的显示和标记
#plt.xticks(())
#plt.yticks(())

plt.show()
