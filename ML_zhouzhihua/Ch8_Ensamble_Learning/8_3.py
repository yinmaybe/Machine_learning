# encoding: utf-8
"""
西瓜数据集3.0alpha 只包含两个连续属性
AdaBoost 集成  未完成
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
