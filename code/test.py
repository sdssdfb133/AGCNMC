import math
from matplotlib import pyplot as plt
from torch import optim
from prepareData import KNearestKnownNeighbors, get_edge_index, read_csv
from model import Model
from trainData import Dataset
from train import adjust_learning_rate, train_model, test_model  
import torch
import random
import csv

class Get_Config(object):
    def __init__(self):
        self.data_path = 'data1'  # 数据集路径
        self.max_iterations = 10  # 循环次数
        self.test = 50  # 测试次数
        self.save_path = 'data1'
        self.epoch = 100
        self.positive_weight = 0.2 # 越接近1则越不敢预测，越接近0则越敢预测
        self.beta1 = 1
        self.beta2 = 1        
        self.K = 5  # 预处理参数
        self.namda = 0.3  # 预处理参数

class Date_Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)  
        self.d = dataset['dd']['data'].size(0)  


def preprocess(K,namda,data_path):
    dataset = dict()
    dataset['md'] = read_csv( data_path +'/m-d.csv')
    dataset['mm'] = read_csv( data_path +'/m-m.csv')
    dataset['dd'] = read_csv( data_path +'/d-d.csv')
    Yd = torch.zeros(dataset['md'].size(0))  # miRNA ##记录每个miRNA与其他的关系用于调整md矩阵
    Yt = torch.zeros(dataset['md'].size(1))  # 疾病 ##记录每个疾病与其他的关系用于调整md矩阵
    miRNA_max = 0
    disease_max = 0
    for i in range(dataset['md'].size(0)):  # 计算一个RNA最多能有多少种病
        if torch.sum(dataset['md'][i]) > miRNA_max:
            miRNA_max = torch.sum(dataset['md'][i])
    for i in range(dataset['md'].size(1)):  ## 最多miRNA
        if torch.sum(dataset['md'].t()[i]) > disease_max:
            disease_max = torch.sum(dataset['md'].t()[i])
    for m in range(dataset['md'].size(0)):  # 处理miRNA矩阵 ## 根据k近邻的miRNA更新Yd
        dnn = KNearestKnownNeighbors(m, dataset['mm'], K)
        w = dict()
        Zd = 0.0
        for i in range(K):
            w[i] = math.pow(namda, i) * dataset['mm'][m][dnn[i]]
            Zd += dataset['mm'][m][dnn[i]]
            Yd[m] += w[i] * (torch.sum(dataset['md'][dnn[i]]) / miRNA_max)
        if K != 0:
            Yd[m] = Yd[m] / Zd
    for d in range(dataset['md'].size(1)):  # 处理疾病矩阵 ## 根据k近邻的疾病更新Yt
        tnn = KNearestKnownNeighbors(d, dataset['dd'], K)
        w = dict()
        Zt = 0.0
        for i in range(K):
            w[i] = math.pow(namda, i) * dataset['dd'][d][tnn[i]]
            Zt += dataset['dd'][d][tnn[i]]
            Yt[d] += w[i] * (torch.sum(dataset['md'].t()[tnn[i]]) / disease_max)
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
    out1 = open('./datasets1/prem-d.csv', 'w', newline='')
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
    dataset['md_train'] = read_csv('./datasets1/prem-d.csv')
    dataset['md_true'] = read_csv('./datasets1/prem-d.csv') # 真实完整的矩阵

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
    with open('./datasets1/result_one.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        one_index = [[int(i) for i in row] for row in reader]
    with open('./datasets1/result_test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        test_index = [[int(i) for i in row] for row in reader]
    ## 这里覆盖了md得到的one_index索引
    print("md_train shape:", dataset['md_train'].shape)

    for i in test_index:    # 对应的predict矩阵的位置设为1，md_ture其实不是真正完整的矩阵了
        dataset['md_train'][i[0]][i[1]] = 0.0             # 修改train矩阵上的1值点，有1/5要变成0值 ## 用于训练
    zero_tensor = torch.LongTensor(zero_index)  # 把1/5的标1(有确定关系)的节点标为0
    one_tensor = torch.LongTensor(one_index)
    test_tensor = torch.LongTensor(test_index)

    ## md结构  
    dataset['md'] = dict() # md的值定义为字典
    dataset['md']['zero'] = zero_tensor
    dataset['md']['one'] = one_tensor
    dataset['md']['test'] = test_tensor
    ##同样更改路径
    dd_matrix = read_csv('./' + opt.data_path + '/d-d.csv')  # dd_matrix放入疾病相似性矩阵，用的是FloatTensor，矩阵大小为435*435
    dd_edge_index = get_edge_index(dd_matrix)   # dd_edge_index是一个多行2列的LongTensor数组，放的是每条边的两端
    dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index} # 键dd对应dd矩阵和边矩阵

    mm_matrix = read_csv('./' + opt.data_path + '/m-m.csv') 
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}
    return dataset


def main():
    opt = Get_Config()
    # 用来存储每轮的最高ROC AUC
    highest_roc_auc_per_iteration = []  # 记录每轮的最高AUC
    best_roc_auc = 0.0  # 全局最佳AUC
    roc_auc = 0.0   # 当前
    patience = 7.0 # 容忍轮数
    X_best, Y_best = None, None  # 初始化X_best和Y_best
    previous_roc_auc = -float('inf')  # 初始化上一次的ROC AUC为负无穷大

    # 读取原始边列表数据
    with open('./datasets/result_data.csv', mode='r') as infile:
        reader = csv.reader(infile)
        edge_list = list(reader)  
    # 打乱数据
    random.shuffle(edge_list)

    # 5折交叉验证
    num_splits = 5
    split_size = len(edge_list) // num_splits

    for fold_idx in range(num_splits):
        print(f"Running fold {fold_idx + 1} of {num_splits}...")

        # 划分数据集为边列表
        test_edges = edge_list[fold_idx * split_size : (fold_idx + 1) * split_size]
        train_edges = edge_list[:fold_idx * split_size] + edge_list[(fold_idx + 1) * split_size:]

        # 保存划分后的数据集
        with open('./datasets/result_test.csv', mode='w', newline='') as outfile_test:
            writer = csv.writer(outfile_test)
            writer.writerows(test_edges)
        with open('./datasets/result_one.csv', mode='w', newline='') as outfile_train:
            writer = csv.writer(outfile_train)
            writer.writerows(train_edges)

        # 准备数据
        dataset = prepare_data(opt)
        train_data = Dataset(opt, dataset)
        sizes = Date_Sizes(dataset)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)

        # 训练和测试逻辑（保持不变）
        highest_roc_auc_this_iteration = 0.0
        previous_roc_auc = -float('inf')
        highest_roc_auc_this_iteration = 0.0  # 用来存储当前轮次的最高ROC AUC
        no_improvement_count = 0.0  # 记录未提升次数
        patience_counter = 0.0 # 下降次数

        for i in range(opt.test):
            train_model(model, train_data[i], optimizer, opt)
            roc_auc, X_best, Y_best, best_roc_auc = test_model(model, train_data[0], X_best, Y_best, best_roc_auc)
            print('dsfs',best_roc_auc)

             # 更新当前轮次的最高ROC AUC
            if roc_auc >= highest_roc_auc_this_iteration:
                highest_roc_auc_this_iteration = roc_auc 
                no_improvement_count = 0.0  # 重置未提升计数
            else:
                no_improvement_count += 1.0  # 未提升计数加1

            # 调整学习率
            patience_counter, no_improvement_count = adjust_learning_rate(
                optimizer, roc_auc, previous_roc_auc, 2, patience_counter, no_improvement_count, highest_roc_auc_this_iteration
            )
            previous_roc_auc = roc_auc

            print(f"Test {i+1} in fold {fold_idx+1} - ROC AUC: {roc_auc}")

            # # 早停检查
            # if no_improvement_count > patience:
            #     print(f"Early stopping at test {i+1} in fold {fold_idx+1}. Best ROC AUC in this fold: {highest_roc_auc_this_iteration}")
            #     break
            print('fold',highest_roc_auc_this_iteration)          
        highest_roc_auc_per_iteration.append(highest_roc_auc_this_iteration)

    # 输出结果（保持不变）
    print("Highest ROC AUC per fold:")
    for i, auc in enumerate(highest_roc_auc_per_iteration, 1):
        print(f"Fold {i}: {auc}")
    print(f"Global Best ROC AUC after all folds: {best_roc_auc}")
    # 计算平均 AUC
    average_auc = sum(highest_roc_auc_per_iteration) / len(highest_roc_auc_per_iteration)
    print(f"\nAverage AUC across all folds: {average_auc:.4f}")

    # 绘制ROC曲线（保持不变）
    if X_best is not None and Y_best is not None:  
        plt.plot(X_best, Y_best, 'k--', label=f'Best ROC (area = {best_roc_auc:.4f})', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Best ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == "__main__":
    main()