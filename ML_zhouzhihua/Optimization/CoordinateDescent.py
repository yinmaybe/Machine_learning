# encoding: utf-8
"""
计算函数 y = -x_1^2 - 3 * x_2^2 + 2 * x_1 * x_2 +6 的最大值
"""
import matplotlib
import numpy as np
import matplotlib.cm as pcm
import matplotlib.mlab as pmlab
import matplotlib.pyplot as plt

delta=0.025
x=np.arange(-3.0,3.0,delta)
y=np.arange(-3.0,3.0,delta)
X,Y=np.meshgrid(x,y)
Z1=-(X**2)
Z2=-(Y**2)
Z=1.0 * (Z1 + 3 * Z2 + 2 * X * Y)+6.0

plt.figure()
plt.contour(X,Y,Z)#画等高线

#取初值
a = []
b = []
a.append(2.0)
b.append(2.0)

j = 1

for i in range(200):
    # 固定y 即b
    a_tmp = b[j-1]
    a.append(a_tmp)
    b.append(b[j-1])

    j = j+1
    #固定x 即a
    b_tmp = a[j-1] / 3
    a.append(a[j-1])
    b.append(b_tmp)

plt.plot(a,b)
max_x1=a[-1]
max_x2=b[-1]

print('当取最大值的时候，x1的取值为：', max_x1)
print('当取最大值的时候，x2的取值为：', max_x2)

print('max f:',-(max_x1**2)-3*(max_x2**2)+2*max_x1*max_x2+6)

plt.title('Coordinate Ascent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()