import numpy as np
def wolf(f, df, p, x0, rho=0.1, sigma=0.5, alpha=1, a=0, b=float("inf"), t=2):
    '''

    :param f: 输入函数
    :param df: 函数梯度
    :param p: 搜索方向,
    :param x0: 初始点
    :param rho: 参数
    :param mu: 参数
    :param alpha:初始步长
    :param a, b: alpha取值区间
    :return: 满足条件的步长
    '''
    flag = 0
    f_0 = f(x0)
    g_0 = df(x0)

    while(flag == 0):
        x_k = x0 + alpha * p
        f_k = f(x_k)
        g_k = df(x_k)
        if (f_k <= f_0 + alpha*rho*np.dot(g_0, p)):
            if (np.dot(g_k, p) >= sigma*np.dot(g_0, p)):
                flag = 1
            # alpha 不够大
            else:
                a = alpha
                if b == float("inf"):
                    alpha = t*alpha
                else:
                    alpha = min(2*alpha, (alpha+b)/2.0)
        # alpha 不够小
        else:
            b = alpha
            alpha = (a + alpha) / 2.0
    return alpha