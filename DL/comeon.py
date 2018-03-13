'''
tensorflow 的简单测试
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 定义层函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))  #随机初始化权重
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)  # 初始化偏置，+0.1
    Ws_plus_b = tf.matmul(inputs,Weights) + biases      # 未使用激活函数的值
    if activation_function is None:
        outputs = Ws_plus_b
    else:
        outputs = activation_function(Ws_plus_b)   # 使用激活函数激活
    return outputs

x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

# None代表无论输入有多少都可以,只有一个特征，所以这里是1
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

 # 第一层，输入层为1，隐含层为10个神经元，Tensorflow 自带的激励函数tf.nn.relu
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

prediction = add_layer(layer1, 10, 1)  # 利用上一层作为输入

# 对二者差的平方求和再取平均
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.ion()  # 绘画之后不暂停,可以在下面继续完成图的交互
plt.show()

for i in range(1000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(ax.lines[0])  # 每次绘画需要移除上次绘画的结果，放在try catch里因为第一次执行没有，所以直接pass
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=3)  # 绘画
        plt.pause(1)  # 停1s
while True:
    plt.show(1)







