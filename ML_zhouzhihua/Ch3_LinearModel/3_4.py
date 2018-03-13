# encoding: utf-8
"""
交叉验证和留一法验证错误率
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.font_manager import FontProperties
from sklearn import linear_model

font = FontProperties(fname=r"C:\Windows\Fonts\simkai.ttf", size=17 )
# 读入csv文件数据
df = pd.read_csv('iris.csv')
all_data = df[["sepal length in cm","sepal width in cm",
         "petal length in cm","petal width in cm",]].values[:,:]
y = np.zeros((150,))
y[0:50] = 1
y[50:100] = 2
y[100:150] = 3
X1 = df[["sepal length in cm","sepal width in cm",
         "petal length in cm","petal width in cm",]].values[0:50,:]
y1 = np.ones((50,))
X2 = df[["sepal length in cm","sepal width in cm",
         "petal length in cm","petal width in cm",]].values[50:100,:]
y2 = np.ones((50,))
y2[:] = 2
X3 = df[["sepal length in cm","sepal width in cm",
         "petal length in cm","petal width in cm",]].values[100:150,:]
y3 = np.ones((50,))
y3[:] =3

#Z[0:15]为第一折  Z[15:20]为第二折  以此类推
def tenfoldsplit(X1, X2, X3):
    Z = np.ones((150, 4))
    z = np.zeros((150,))
    i = 0
    k = 0
    for j in range(10):
        Z[i:i+5,:] = X1[k:k+5,:]
        z[i:i+5] = 1
        Z[i+5:i+10,:] = X2[k:k+5,:]
        z[i+5:i+10] = 2
        Z[i+10:i+15,:] =X3[k:k+5,:]
        z[i+10:i+15] = 3
        i = i + 15
        k = k + 5
    return Z,z

Z, z = tenfoldsplit(X1, X2, X3)
lr = linear_model.LogisticRegression(C=1e5)

def cal_error(lr, test_data, test_label):
    y = lr.predict(test_data)
    sum = 0
    for i in range(15):
        if y[i]!= test_label[i]:
            sum += 1
    error = sum/15.0
    return error

def tenfold_cross_validation(Z, z, lr):
    train_data = np.zeros((135, 4))
    train_label = np.zeros((135,))
    test_data = np.zeros((15, 4))
    test_label = np.zeros((15,))
    error = np.zeros((10,))
    for i in range(10):
        test_data[0:15] = Z[i*15:(i+1)*15]
        test_label[0:15] = z[i*15:(i+1)*15]
        if i!=0:
            train_data[0:i*15] = Z[0:i*15]
            train_label[0:i*15] = z[0:i*15]
        if i!=9:
            train_data[i*15:] = Z[(i+1)*15:]
            train_label[i*15:] = z[(i+1)*15:]
        lr.fit(train_data, train_label)
        error[i] = cal_error(lr,test_data, test_label)
    error_mean = np.mean(error)
    return error_mean

#十折交叉验证法错误率

print("10折交叉验证法error：", tenfold_cross_validation(Z,z,lr))

#留一法
def lou(Z,z,lr):
    train_data = np.zeros((149, 4))
    train_label = np.zeros((149,))
    test_data = np.zeros((1, 4))
    sum = 0
    for i in range(150):
        test_data[0] =Z[i]
        test_label = z[i]
        if i!= 0:
            train_data[0:i] = Z[0:i]
            train_label[0:i] = z[0:i]
        if i!= 149:
            train_data[i:] = Z[i+1:]
            train_label[i:] = z[i+1:]
        lr.fit(train_data, train_label)
        cla = lr.predict(test_data)
        if cla[0]!=test_label:
            sum += 1
    error_mean = sum/150.0
    return error_mean

#留一法错误率
print("留一法error：",lou(Z, z, lr) )

# sklearn.cross_validation 将会被遗弃
# 使用sklearn.model_select


from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn import metrics
iris = datasets.load_iris()
lr = linear_model.LogisticRegression(C=1e5)
predicted = cross_val_predict(lr, iris.data, y=iris.target, cv=10)
score = metrics.accuracy_score(iris.target, predicted)
error = 1 - score
print("使用sklearn测出的10折交叉验证错误率：", error)

# 留一
loo = LeaveOneOut()
lr = linear_model.LogisticRegression(C=1e5)
sum = 0
# 看源码 index选取好像是固定的不是随机的，有需要的话可以先shuffle
for train_index, test_index in loo.split(all_data):
    X_train_data, X_test_data = all_data[train_index], all_data[test_index]
    y_train_label, y_test_label = y[train_index], y[test_index]
    lr.fit(X_train_data, y_train_label)
    predicted = lr.predict(X_test_data)
    if predicted!= y_test_label:
        sum += 1
error = sum/150.0
print("使用sklearn测出的留一法错误率：", error)







