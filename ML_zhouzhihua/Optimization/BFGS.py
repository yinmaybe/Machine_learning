# encoding: utf-8
from Optimization.LinearSearch import wolf
from scipy.optimize import fmin_bfgs
import numpy as np
from scipy.linalg import norm
from math import sqrt

# 简单的bfgs算法实现
"""
BFGS的收敛速度慢于牛顿法
"""

def bfgs(func, jac, x0, eps=1e-5):
    g0 = jac(x0)
    m = len(g0)
    H0 = np.identity(m)
    x0 = x0
    num_ite = 0
    while(norm(g0)>eps):
        num_ite += 1
        pk = -np.dot(H0, g0)
        alpha = wolf(func, jac, pk, x0)
        xk = x0 + alpha*pk
        sk = xk - x0
        yk = jac(xk) - g0

        wk = sqrt(np.dot(np.dot(H0, yk), yk))*((float(1)/np.dot(yk, sk))*sk - (float(1)/np.dot(np.dot(H0, yk), yk))*np.dot(H0, yk))
        Hk = H0 - np.outer(np.dot(H0, yk), np.dot(H0.T, yk))/float(np.dot(np.dot(H0, yk), yk)) + \
                                           np.outer(sk, sk)/float(np.dot(sk, yk)) + np.dot(wk, wk)
        # 没有matlab写的方便
        #Hk = H0 + float(1)/np.dot(sk, yk)*(1 + float(np.dot(np.dot(H0, yk), yk))/np.dot(sk, yk))*np.outer(sk, sk)  - \
        #      float(1)/np.dot(sk, yk)*(np.dot(np.outer(sk, yk), H0) + np.outer(np.dot(H0, yk),sk))

        x0 = xk
        g0 = jac(xk)
        H0 = Hk
    return x0, num_ite

# 最优点是(1, 1)
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1 - x[0])**2

def jacobian(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])

x = np.array([4, 5])
x_opt, num_iter = bfgs(rosenbrock, jacobian, x)
# 迭代次数与设置的允许误差有关
print("自己写的bfgs所求最优解:", x_opt, "  ", "迭代次数：", num_iter)
print("\n\n")
print("scipy库所求结果：")
print(fmin_bfgs(rosenbrock, x, fprime=jacobian))