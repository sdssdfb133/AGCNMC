from matplotlib import pyplot as plt
from torch import optim
from prepareData import prepare_data
from model import Model
from trainData import Dataset
from train import train_model, test_model, adjust_learning_rate
import torch

class Get_Config(object):
    def __init__(self):
        self.data_path = 'data'  # 数据集路径
        self.max_iterations = 10  # 循环次数
        self.test = 50  # 测试次数
        self.save_path = 'data'
        self.epoch = 100
        self.positive_weight = 0.2 # 越接近1则越不敢预测，越接近0则越敢预测
        self.beta1 = 1
        self.beta2 = 1        
        self.K = 1  # 预处理参数
        self.namda = 0.3  # 预处理参数

class Date_Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['data'].size(0)  
        self.d = dataset['dd']['data'].size(0)  

def main():
    opt = Get_Config()
    # 用来存储每轮的最高ROC AUC
    highest_roc_auc_per_iteration = []  # 记录每轮的最高AUC
    best_roc_auc = 0.0  # 全局最佳AUC
    roc_auc = 0.0   # 当前
    patience = 7.0 # 容忍轮数
    X_best, Y_best = None, None  # 初始化X_best和Y_best
    previous_roc_auc = -float('inf')  # 初始化上一次的ROC AUC为负无穷大

    for j in range(opt.max_iterations):
        dataset = prepare_data(opt)
        train_data = Dataset(opt, dataset)
        sizes = Date_Sizes(dataset)
        model = Model(sizes)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
        # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)  # 恒定学习率
        highest_roc_auc_this_iteration = 0.0  # 用来存储当前轮次的最高ROC AUC
        no_improvement_count = 0.0  # 记录未提升次数
        patience_counter = 0.0 # 下降次数
        for i in range(opt.test):
            train_model(model, train_data[i], optimizer, opt)
            roc_auc, X_best, Y_best, best_roc_auc = test_model(model, train_data[0], X_best, Y_best, best_roc_auc)
            
            # 更新当前轮次的最高ROC AUC
            if roc_auc >= highest_roc_auc_this_iteration:
                highest_roc_auc_this_iteration = roc_auc 
                no_improvement_count = 0.0  # 重置未提升计数
            if roc_auc < highest_roc_auc_this_iteration: 
                if roc_auc>previous_roc_auc:
                    no_improvement_count += 0.5
                else:
                    no_improvement_count += 1.0     
            else:
                no_improvement_count += 1.0  # 未提升计数加1
            print('0',no_improvement_count)           # 调整学习率    
            patience_counter ,no_improvement_count= adjust_learning_rate(optimizer, roc_auc, previous_roc_auc, 2, patience_counter, no_improvement_count,highest_roc_auc_this_iteration)
            previous_roc_auc = roc_auc
            print(f"Test {i+1} in iteration {j+1} - ROC AUC: {roc_auc}")
            # 检查是否需要早停
            if no_improvement_count > patience:
                print(f"Early stopping at test {i+1} in iteration {j+1}. Best ROC AUC in this iteration: {highest_roc_auc_this_iteration}")
                break

        # 保存当前轮次的最高ROC AUC到列表
        highest_roc_auc_per_iteration.append(highest_roc_auc_this_iteration)

        # 打印当前轮次的最高ROC AUC
        print(f"Highest ROC AUC in iteration {j+1}: {highest_roc_auc_this_iteration}")
        print(f"Best ROC AUC after all iterations: {best_roc_auc}")
    # 输出每轮训练后的最高AUC
    print("Highest ROC AUC per iteration:")
    for i, auc in enumerate(highest_roc_auc_per_iteration, 1):
        print(f"Iteration {i}: {auc}")
    # 输出全局最佳AUC
    print(f"Global Best ROC AUC after all iterations: {best_roc_auc}")

    # 最终绘制全局最佳ROC曲线
    if X_best is not None and Y_best is not None:  
        plt.plot(X_best, Y_best, 'k--', label=f'Best ROC (area = {best_roc_auc:.4f})', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Best ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == "__main__":
    main()
