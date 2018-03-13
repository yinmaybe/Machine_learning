# encoding; utf-8
import numpy as np
import scipy.optimize
import pandas as pd
from math import log2
from collections import  deque

"""
ID3序选取熵的增益最为指标，偏向于选取属性值多的属性，如：若每个样本有一个单独编号，则编号是最佳属性
熵的增益率作为指标，偏向于选取属性值少的属性，
ID4.5先选取候选的高信息增量熵的几个属性，在从中选取增益率最高的
CART树选取基尼指数最为指标

其实这些指标的影响不大，但剪枝的影响很大，剪枝需要将数据分为训练集和验证集。

剪枝是为了防止树的过拟合，分为预剪枝和后剪枝
预剪枝或造成欠拟合，但计算量小于后剪枝
后剪枝的效果相对更好

缺失值的处理
"""
# 剪枝的部分不想写了， 觉得直接使用sklearn就可以， 花点时间在优化算法上分

df = pd.read_csv(r"../watermelon.csv", encoding="utf-8")
dataSet = df.values[0:,1:].tolist()  #去除编号那列
attr_list = {"色泽":[0, True,True], "根蒂":[1, True, True], "敲声":[2, True, True],
                "纹理":[3, True, True], "脐部":[4, True, True], "触感":[5, True, True],
                    "密度":[6, True, False], "含糖率":[7, True, False]}
# 计算香农熵
def calShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log2(prob)
    return shannonEnt

# 计算以指定列属性划分后的信息增益熵， col_number是列属性的index
# 属性值是离散的或连续的 若是离散的还返回value
def cal_col_ent(dataSet, col_number, continues = False):
    m = len(dataSet)
    shannonEnt_pre = calShannonEnt(dataSet)
    value = None
    if  continues:
        valuelist = []
        for i in range(m):
            valuelist.append(dataSet[i][col_number])

        #sortvaluelist = valuelist.sort()   原地排序   无返回值
        sortvaluelist = sorted(valuelist)
        splitlist = []
        for i in range(len(valuelist) - 1):
            splitlist.append((sortvaluelist[i] + sortvaluelist[i+1])/2)
        shannonEnt_aft = 10000
        for splitvalue in splitlist:
            list_lt = []
            list_ge = []
            col_label = {"lt": 0, "ge": 0}
            for i in range(m):
                if dataSet[i][col_number] < splitvalue:
                    col_label["lt"] += 1
                    list_lt.append(dataSet[i])
                else:
                    col_label["ge"] += 1
                    list_ge.append(dataSet[i])
            a = calShannonEnt(list_lt)
            b = calShannonEnt(list_ge)
            c = (float(col_label["lt"])/m)*a + (float(col_label["ge"])/m)*b
            if shannonEnt_aft > c:   #选取最小的shannonEnt_aft
                shannonEnt_aft = c
                value = splitvalue
        return shannonEnt_pre - shannonEnt_aft, value
    else:
        # col_label 存储列属性的取值 {“label_1_name”: sum_value,  }
        col_label = {}
        for i in range(m):
            if dataSet[i][col_number] not in col_label.keys():
                col_label[dataSet[i][col_number]] = 1
            else:
                col_label[dataSet[i][col_number]] += 1
        ll = len(col_label)
        shannonEnt_aft = 0
        for key in col_label.keys():
            list = []
            for row in range(m):
                if dataSet[row][col_number] == key:
                    list.append(dataSet[row])
            shannonEnt_aft = shannonEnt_aft + (col_label[key]/float(m))*calShannonEnt(list)

        return shannonEnt_pre - shannonEnt_aft

# 选择最佳划分属性, 并按该属性划分成多个数据子集
# attr_list 是一个字典，结构是{"列属性名"：[列索引, Ture(or False)（是否可以作为划分属性）,Ture（of False)（Ture是离散属性） ] }
# 返回多个子集  最佳划分属性  若该属性是连续的，返回划分点
def selcet_best_attr(dataSet, attr_list):
    m = len(dataSet)
    n = len(dataSet[0])
    gainEnt = -1000
    best_attr = None
    value = None
    for key in attr_list:
        if attr_list[key][1]:
            if attr_list[key][2]: # 是离散属性
                x = cal_col_ent(dataSet, attr_list[key][0] )
            else:
                # 出bug 最佳连续属性value会被后面的连续属性计算出的value盖
                #x, value = cal_col_ent(dataSet, attr_list[key][0], continues=True)
                x, value1 = cal_col_ent(dataSet, attr_list[key][0], continues=True)
            if x>gainEnt:
                gainEnt = x
                best_attr = key
                if not attr_list[key][2]:
                    value = value1

    if  attr_list[best_attr][2]:  # 是离散属性,该属性在子集中不能再作为划分属性
        attr_list[best_attr][1] = False

    # 按该属性划分子集
    labellist = []
    col = attr_list[best_attr][0]
    data_ret = []
    if attr_list[best_attr][2]:  #划分属性是离散的
        for row in range(m):
            if dataSet[row][col] not in labellist:
                labellist.append(dataSet[row][col])
        i = 0
        for key in labellist:
            data_ret.append([])
            for row in range(m):
                if dataSet[row][col] == key:
                    data_ret[i].append(dataSet[row])
            i += 1

    else:
        labellist = ["<", ">="]
        data_ret.append([])
        data_ret.append([])
        for row in range(m):
            if dataSet[row][col]<value:
                data_ret[0].append(dataSet[row])
            else:
                data_ret[1].append(dataSet[row])
    if attr_list[best_attr][2]:
        value = None
    return data_ret, attr_list,labellist, best_attr, value


# 标签相同 不可分
# 所有属性值相同 不可分
# 返回值：split  bool
def is_cont_split(subdata):
    l = len(subdata)
    c = len(subdata[0]) #列数
    split = True
    for i in range(l):
        num1 = 0
        num2 = 0
        label = subdata[0][c - 1]
        one_row = subdata[0]
        for row in range(l):
            if subdata[row] == one_row:
                num1 += 1
            if subdata[row][c - 1] == label:
                num2 += 1
        if num1 == l or num2 == l:
            # 不可分 标明类别
            split = False
    return split

# 对数据集按属性划分，返回同一类样本个数最多的类标签
def majorvote(dataSet):
    m = len(dataSet)
    n = len(dataSet[0])
    labeldict = {}
    for i in range(m):
        if dataSet[i][n-1] not in labeldict:
            labeldict[dataSet[i][n-1]] = 1
        else:
            labeldict[dataSet[i][n - 1]] += 1
    numb = -1
    label = None
    for key in labeldict.keys():
        if labeldict[key]>numb:
            numb = labeldict[key]
            label = key
    return label


# 当树的层数太深时，应当避免使用递归，防止“栈”溢出
# 用字典保存划分的树
def create_ID3_tree(dataSet, attr_list):
    header = {}
    deq_mytree = deque()
    deq_mytree.append(header)
    deq = deque()
    deq.append(dataSet)
    deq_attrlist = deque()
    deq_attrlist.append(attr_list)
    while(len(deq)!=0):
        data = deq.pop()
        attr_list = deq_attrlist.pop()

        # 判断是否是叶节点
        mytree = deq_mytree.pop()
        num = 0
        for i in attr_list.keys():
            if not attr_list[i][1]:
                num += 1
        split = is_cont_split(data)
        if num == len(attr_list) or not split:
            mytree["不可分"] = majorvote(data)
            continue

        data_ret, attr_list, labellist, best_attr, value = selcet_best_attr(data, attr_list)
        # 深度优先， 实际使用时可设置最大深度
        #若要改为广度优先，可将栈改为队列
        m = len(labellist)
        if not attr_list[best_attr][2]:            # 划分属性是连续属性
            name = best_attr + "<"+str(value)
            mytree[name] = {"是":{}, "否":{}}
            deq_mytree.append(mytree[name]["是"])
            deq_mytree.append(mytree[name]["否"])
        else:

            mytree[best_attr] = {}
            for i in range(m):
                mytree[best_attr][labellist[i]] = {}
                deq_mytree.append(mytree[best_attr][labellist[i]])

        for i in range(m):
            deq.append(data_ret[i])
            deq_attrlist.append(attr_list)
    return header

print(create_ID3_tree(dataSet, attr_list))



"""
1.  计算未划分是的香农熵  ent = calcShannonEnt(dataSet)
2.  选择一个最佳划分属性，返回按该属性划分后的多个子数据集和一个字典，用字典保存划分结果
    {   for 每一个可以划分的属性
        {
            判断该属性是离散属性或连续属性
            计算增益熵
        }
        选择增益熵最大的属性，若该属性是离散属性则将该属性在子数据集中不可划分
    }
3.  判断该子数据集是否不需要划分，若仍需要划分，继续酸则最佳划分属性，重复第二步

若不要递归，可以使用栈或队列进行深度优先或广度优先划分属性
"""