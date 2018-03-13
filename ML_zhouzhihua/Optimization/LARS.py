# encoding: utf-8
import numpy as np
from sklearn.preprocessing import normalize as f_normalize
from scipy.linalg import norm
from scipy.linalg import inv
from math import sqrt

# 数据标准化
def _normalize(X, y):
    X_offset = np.mean(X, axis=0)
    X = X - X_offset
    y_offset = np.average(y, axis=0)
    y = y - y_offset
    X, X_scale = f_normalize(X, axis=0, return_norm=True)
    return X, y, X_offset, y_offset, X_scale

def lars(X, y, max_iter=100, eps = 1e-5):
    X0 = X
    y0 = y
    X, y, X_offset, y_offset, X_scale = _normalize(X, y)
    n_features = X.shape[1]
    n_samples = y.size
    # 初始化
    mu = np.zeros(n_samples, dtype="float32")  #逼近值
    re = y - mu  #残差向量
    red = norm(re)
    max_features = min(max_iter, n_features)
    active = list()
    # holds the sign of covariance
    sign_active = np.zeros(max_features, dtype=np.int8)
    # 添加一个初始的活跃集
    Cov = np.dot(X.T, re)
    C_idx = np.argmax(np.abs(Cov))
    active.append(C_idx)
    active.sort()
    beta = np.zeros((n_samples,))  #存储系数
    while red>eps or len(active)<=max_features:
        # 相关系数
        Cov = np.dot(X.T, re)
        for i in range(len(Cov)):
            if Cov[i] < 0:
                sign_active[i] = -1
            else:
                sign_active[i] = 1
        C_idx = np.argmax(np.abs(Cov))
        Cov_value = np.max(np.abs(Cov))
        # 计算角平分线，即前进方向
        l = len(active)
        X_active = np.zeros((n_samples, l))
        for i in range(l):
            X_active[:,i] = sign_active[i]*X[:, active[i]]
        G_active = np.dot(X_active.T, X_active)
        G_inv = inv(G_active)
        one_vector = np.ones((l,))
        A_active = float(1)/sqrt(np.dot(one_vector, np.dot(G_inv, one_vector)))
        w_active = A_active * np.dot(G_inv, one_vector)
        mu_active = np.dot(X_active, w_active)  # 前进方向
        #计算前进步长
        gamma1 = -1
        gamma2 = -1
        ind1 = -1
        ind2 = -1
        step = list()
        ind1_list = list()
        step_ = list()
        ind2_list = list()
        if l<n_features:
            a = np.dot(X.T, mu_active)
            for i in range(n_features):
                if i not in active:
                    tmp1 = (Cov_value - Cov[i])/(A_active - a[i])
                    tmp2 = (Cov_value + Cov[i])/(A_active + a[i])
                    if tmp1>0:
                        step.append(tmp1)
                        ind1_list.append(i)
                    if tmp2>0:
                        step.append(tmp2)
                        ind1_list.append(i)
            gamma1 = min(step)
            ind1 = ind1_list[step.index(gamma1)]
        else:
            gamma1 = Cov_value/A_active
        index = 0
        for i in active:
            if w_active[index]!=0:
                tmp3 = -beta[i]/(sign_active[i]*w_active[index])
                index += 1
                if tmp3>0:
                    step_.append(tmp3)
                    ind2_list.append(i)
        if len(step_)==0:
            gamma2 = -1
        else:
            gamma2 = min(step_)
            ind2 = ind2_list[step_.index(gamma2)]
        # 判断是否违反条件
        if gamma2 < gamma1 and gamma2 > 0 :
            mu += gamma2 * mu_active
            #这是不多的  角平分线与投影不一样
            """
            beta_active = gamma2*np.dot(X_active.T, mu_active)
            for i in range(len(active)):
                beta[active[i]] += beta[active[i]] + beta_active[i]
            """
            for i in range(len(active)):
                beta[active[i]] = beta[active[i]] + gamma2*w_active[i]
            re = y - mu
            red = norm(re)
            active.remove(ind2)
        else:
            mu += gamma1 * mu_active
            """
            beta_active = gamma1 * np.dot(X_active.T, mu_active)
            for i in range(len(active)):
                beta[active[i]] = beta[active[i]] + beta_active[i]
            """
            for i in range(len(active)):
                beta[active[i]] = beta[active[i]] + gamma1*w_active[i]
            re = y - mu
            red = norm(re)
            active.append(ind1)
            active.sort()

        print(beta)
    # 计算真正的beta 和 intercept
    beta = beta / X_scale
    b = np.mean(y0 - np.dot(X0, beta))
    print("真正的beta值和截距值：")
    print(beta)
    print(b)


# test

X = np.array([[-1, 2, 6], [1, -5, -8], [1, 0, 9]] )
y = np.array([-2, 4, 7])
lars(X, y)

# 内置库  结果是相同的
from sklearn import linear_model
reg = linear_model.Lars()
reg.fit(X, y)
print("\n\n")
print(reg.coef_, "\t", reg.intercept_)
print(reg.coef_path_)