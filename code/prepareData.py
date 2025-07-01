import csv
import math

import torch as t
import random
from operator import itemgetter

def read_csv(path):
    with open(path, 'r', newline='') as csv_file: # open的时候会自动\n，newline=''可以避免这种情况
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader] # 读入整个邻接矩阵
        return t.FloatTensor(md_data)   # 转化为FloatTensor类型，其实就是32位float型的二维数组


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:   # 如果边权不为0，则把两个节点分别加入edge_index
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)
## 提取非零元素构建edge_index

def KNearestKnownNeighbors(d, Smatrix, K):  # 传入d从0开始
    # k_neighbors = []
    # value_list = []
    # for i in range(Smatrix.size(1)):
    #     value_list.append([Smatrix[d][i], i])
    # value_list = sorted(value_list, key=itemgetter(0), reverse=True)
    # for i in range(K):
    #     k_neighbors.append(value_list[i][1])
    # return k_neighbors
     # 获取d行的相似度向量，torch.topk会自动返回最大的K个值和索引
    distances = Smatrix[d]  # 获取第d行相似度向量
    top_k_values, top_k_indices = t.topk(distances, K, largest=True, sorted=False)
    
    # 返回K个邻居节点的索引
    return top_k_indices.tolist()  # 转换为列表形式返回
## 寻找d的k近邻按相似度

def preprocess(K,namda,data_path):
    dataset = dict()
    dataset['md'] = read_csv( data_path +'/m-d.csv')
    dataset['mm'] = read_csv( data_path +'/m-m.csv')
    dataset['dd'] = read_csv( data_path +'/d-d.csv')
    Yd = t.zeros(dataset['md'].size(0))  # miRNA ##记录每个miRNA与其他的关系用于调整md矩阵
    Yt = t.zeros(dataset['md'].size(1))  # 疾病 ##记录每个疾病与其他的关系用于调整md矩阵
    miRNA_max = 0
    disease_max = 0
    for i in range(dataset['md'].size(0)):  # 计算一个RNA最多能有多少种病
        if t.sum(dataset['md'][i]) > miRNA_max:
            miRNA_max = t.sum(dataset['md'][i])
    for i in range(dataset['md'].size(1)):  ## 最多miRNA
        if t.sum(dataset['md'].t()[i]) > disease_max:
            disease_max = t.sum(dataset['md'].t()[i])
    for m in range(dataset['md'].size(0)):  # 处理miRNA矩阵 ## 根据k近邻的miRNA更新Yd
        dnn = KNearestKnownNeighbors(m, dataset['mm'], K)
        w = dict()
        Zd = 0.0
        for i in range(K):
            w[i] = math.pow(namda, i) * dataset['mm'][m][dnn[i]]
            Zd += dataset['mm'][m][dnn[i]]
            Yd[m] += w[i] * (t.sum(dataset['md'][dnn[i]]) / miRNA_max)
        if K != 0:
            Yd[m] = Yd[m] / Zd
    for d in range(dataset['md'].size(1)):  # 处理疾病矩阵 ## 根据k近邻的疾病更新Yt
        tnn = KNearestKnownNeighbors(d, dataset['dd'], K)
        w = dict()
        Zt = 0.0
        for i in range(K):
            w[i] = math.pow(namda, i) * dataset['dd'][d][tnn[i]]
            Zt += dataset['dd'][d][tnn[i]]
            Yt[d] += w[i] * (t.sum(dataset['md'].t()[tnn[i]]) / disease_max)
        if K != 0:
            Yt[d] = Yt[d] / Zt
    for i in range(dataset['md'].size(0)): ## 更新md
        for j in range(dataset['md'].size(1)):
            if (Yd[i]+Yt[j])/2 >=1:
                print('too big')
                dataset['md'][i][j] = max(0.92,dataset['md'][i][j])
            else:
                dataset['md'][i][j] = max((Yd[i] + Yt[j]) / 2, dataset['md'][i][j])
    
    ## 保存为 prem-d.csv文件
    out1 = open('./datasets/prem-d.csv', 'w', newline='')
    csv_write1 = csv.writer(out1, dialect="excel")
    for i in range(dataset['md'].size(0)):
        csv_write1.writerow([i.item() for i in dataset['md'][i]])

def prepare_data(opt):
    preprocess(opt.K,opt.namda,opt.data_path)
    dataset = dict()        # dataset被定义为字典
    # dataset['md_train'] = read_csv(opt.data_path + '\\m-d.csv') # data path就是../data，\\是转义字符，也即\  注意此处md_p才是真正的矩阵
    # dataset['md_true'] = read_csv(opt.data_path + '\\m-d.csv') # 真实完整的矩阵
    ## 这更改了路径使其在整个代码文件使用
    ## 读取prem-d.csv 文件作为训练数据和真实数据
    dataset['md_train'] = read_csv('./datasets/prem-d.csv')
    dataset['md_true'] = read_csv('./datasets/prem-d.csv') # 真实完整的矩阵

    zero_index = []
    one_index = []
    test_index=[]
    for i in range(dataset['md_true'].size(0)):  # .size(0)是获得FloatTensor矩阵的行
        for j in range(dataset['md_true'].size(1)):  # .size(1)是获得FloatTensor矩阵的列

            if dataset['md_true'][i][j] < 1:   # 小于1，其实就是0.0，就计入zero_index这个列表
                zero_index.append([i, j])
            elif dataset['md_true'][i][j] >= 1:   # 大于等于1，其实就是1.0，就计入one_index这个列表
                one_index.append([i, j])
    ## 索引标记正负样本

    # random.shuffle(one_index)
    # random.shuffle(zero_index)
    # 控制变量进行对照实验
    ## 这同样更改了路径使其在整个代码文件使用
    with open('./datasets/result_one.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        one_index = [[int(i) for i in row] for row in reader]
    with open('./datasets/result_test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        test_index = [[int(i) for i in row] for row in reader]
    ## 这里覆盖了md得到的one_index索引

    for i in test_index:    # 对应的predict矩阵的位置设为1，md_ture其实不是真正完整的矩阵了
        dataset['md_train'][i[0]][i[1]] = 0.0             # 修改train矩阵上的1值点，有1/5要变成0值 ## 用于训练
    zero_tensor = t.LongTensor(zero_index)  # 把1/5的标1(有确定关系)的节点标为0
    one_tensor = t.LongTensor(one_index)
    test_tensor = t.LongTensor(test_index)

    ## md结构  
    dataset['md'] = dict() # md的值定义为字典
    dataset['md']['zero'] = zero_tensor
    dataset['md']['one'] = one_tensor
    dataset['md']['test'] = test_tensor
    ##同样更改路径
    dd_matrix = read_csv( opt.data_path + '/d-d.csv')  # dd_matrix放入疾病相似性矩阵，用的是FloatTensor，矩阵大小为435*435
    dd_edge_index = get_edge_index(dd_matrix)   # dd_edge_index是一个多行2列的LongTensor数组，放的是每条边的两端
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index} # 键dd对应dd矩阵和边矩阵

    mm_matrix = read_csv( opt.data_path + '/m-m.csv') # 757*757的mm矩阵
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset

