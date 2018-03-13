import random
"""
可以看出批量梯度下降比随机梯度下降更接近最优值，但批量随机梯度下降每次迭代时的计算复杂度也高一些
迭代次数：有时批量梯度下降次数多，有时随机梯度下降迭代次数多
"""
# matrix_A  训练集
matrix_A = [[1,4], [2,5], [5,1], [4,2]]
Matrix_y = [19,26,19,20]
theta = [2,5]
#学习速率
leraing_rate = 0.005
loss = 50
iters = 1
Eps = 0.0001
#随机梯度下降
while loss>Eps and iters <1000 :
    loss = 0
    i = random.randint(0, 3)
    h = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1]
    #theta的每次更新只需要一个样本点
    theta[0] = theta[0] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][0]
    theta[1] = theta[1] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][1]
    Error = 0
    Error = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1] - Matrix_y[i]
    Error = Error*Error
    loss = loss +Error
    iters = iters +1
print("随机梯度下降：")
print ('theta=',theta)
print ('iters=',iters)
print("\n\n")


theta = [2,5]
loss = 50
iters = 1
Eps = 0.0001
#梯度下降
while loss>Eps and iters <1000 :
    loss = 0
    #theta的一次迭代需要遍历所有样本点
    for i in range(4):
        h = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1]
        theta[0] = theta[0] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][0]
        theta[1] = theta[1] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][1]
    for i in range(4):
        Error = 0
        Error = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1] - Matrix_y[i]
        Error = Error*Error
        loss = loss +Error
    iters = iters +1
print("完整的梯度下降：")
print ('theta=',theta)
print ('iters=',iters)
print("\n\n")


theta = [2,5]
loss = 50
iters = 1
Eps = 0.0001
#批量梯度下降
while loss>Eps and iters <1000 :
    loss = 0
    # theta的一次迭代需要两个样本点
    sampleindex =  random.sample([0,1,2,3],2)
    for i in sampleindex :
        h = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1]
        theta[0] = theta[0] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][0]
        theta[1] = theta[1] + leraing_rate*(Matrix_y[i]-h)*matrix_A[i][1]
    for i in sampleindex :
        Error = 0
        Error = theta[0]*matrix_A[i][0] + theta[1]*matrix_A[i][1] - Matrix_y[i]
        Error = Error*Error
        loss = loss +Error
    iters = iters +1
print("批量梯度下降：")
print ('theta=',theta)
print ('iters=',iters)
