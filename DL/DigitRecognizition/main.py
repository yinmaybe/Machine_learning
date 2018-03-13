"""
书中非完整代码，非个人原创，只是个人觉得很好，完整代码可参考
https://github.com/mnielsen/neural-networks-and-deep-learning
<<Neural Networks and Deep Learning>>
http://neuralnetworksanddeeplearning.com/
"""

import mnist_loader
import network
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


'''
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
'''

net = network2.Network([784, 30, 10],  cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)
