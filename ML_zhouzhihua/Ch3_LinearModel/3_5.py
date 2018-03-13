# -*- coding:utf-8 -*-
"""
LDA 分类
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from  matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simkai.ttf", size=17 )
file1 = open(r'../watermelon.csv','r',encoding="utf-8")
data = [line.strip('\n').split(',') for line in file1]
X = [[float(raw[-3]), float(raw[-2])] for raw in data[1:]]
y = [1 if raw[-1].strip()=='是' else 0 for raw in data[1:]]
X = np.array(X)
y = np.array(y)
#######################################################################以上是西瓜

# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

###############################################################################
# plot functions
def plot_data(lda, X, y, y_pred):
    plt.figure()
    plt.title('线性判别分析', fontproperties=font)
    plt.xlabel('含糖率', fontproperties=font)
    plt.ylabel('密度', fontproperties=font)
    tp = (y == y_pred)  # True Positive //Boolean matrix

    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    # 画出概率值为0.5的等值线
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

###############################################################################
# Linear Discriminant Analysis
# 还有二次线性判别 QDA
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
plot_data(lda, X, y, y_pred)
#plt.axis('tight')
plt.show()