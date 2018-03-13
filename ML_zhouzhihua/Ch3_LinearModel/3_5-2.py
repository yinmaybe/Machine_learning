# encoding: utf-8
"""
LDA 分类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\Windows\Fonts\simkai.ttf", size=17 )
# 读入csv文件数据
df = pd.read_csv(r'../watermelon.csv', encoding="utf-8",skipinitialspace=True)
# 常理来说矩阵中是不能含有非数值型数据的
dataMat = np.mat(df[["密度", "含糖率", "好瓜"]].values[:,:])
index = dataMat[:,2] == "是"
m, n = dataMat.shape
index = np.ones((m, n ))
for i in range(m):
    if(dataMat[i,2] == "是"):
        index[i,:] = 0
#正例
postive = dataMat[index == 0].reshape(-1,3)
#反例
negative = dataMat[index == 1].reshape(-1,3)

ps_data = postive[:,[0,1]]
nt_data = negative[:, [0,1]]

#计算均值
mu1 = np.mean(ps_data, axis=0)
mu0 = np.mean(nt_data, axis=0)

#重新计算正反例特征减去均值的数据
m,n = ps_data.shape
ps_data_std = np.zeros((m, n))
for i in range(n):
    for j in range(m):
        ps_data_std[j, i] = ps_data[j, i] - mu1[0,i]

m,n = nt_data.shape
nt_data_std = np.zeros((m, n))
for i in range(n):
    for j in range(m):
        nt_data_std[j, i] = nt_data[j, i] - mu0[0,i]

#计算Sw和其逆
Sw = np.dot(ps_data_std.T, ps_data_std) + np.dot(nt_data_std.T, nt_data_std)
Sw_inv = np.linalg.inv(Sw)

# 计算权重
w = np.dot(Sw_inv, (mu0 - mu1).T)


#预测 输入x是矩阵(n_samples, n_features) ,输出是列向量
#判断准则是比较 样本点投影后所得点 和 类中心点投影后的点 的距离，与谁近，预测为该类
def predict(x):
    m, n = x.shape
    z = np.dot(x,w)
    a = np.dot(mu0,w)
    b = np.dot(mu1, w)
    y = np.zeros((m, 1))
    for i in range(m):
        if abs(z[i,0] - a)>abs(z[i,0] - b):
            y[i, 0] = 1
        if abs(z[i,0] - a) == abs(z[i,0] - b):
            y[i, 0] = 0.5  #表示0,1的可能性相等 即分界线
    return y

# 画图
#密度最小值 最大值
xmin = min(dataMat[:, 0]) - 0.1
xmax = max(dataMat[:, 0]) + 0.1
# 含糖率最小值 最大值
ymin = min(dataMat[:, 1]) - 0.1
ymax = max(dataMat[:, 1]) + 0.1

h = 0.005

xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
test = np.c_[xx.ravel(), yy.ravel()]
Z = predict(test).reshape(xx.shape)


index_p = predict(ps_data).ravel()

# 画正例 区别真正例和假反例
plt.plot(ps_data[index_p==1, 0], ps_data[index_p==1, 1], "o", color="red")
plt.plot(ps_data[index_p==0, 0], ps_data[index_p==0, 1], "+", color="red")

index_n = predict(nt_data).ravel()

# 画反例
plt.plot(nt_data[index_n==0, 0],nt_data[index_n==0, 1], "o", color="blue")
plt.plot(nt_data[index_n==1, 0],nt_data[index_n==1, 1], "+", color="blue")

plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')
plt.xlabel('密度', fontproperties = font )
plt.ylabel('含糖率', fontproperties = font)
plt.title("线性判别", fontproperties = font)

# 效果还行
plt.show()







