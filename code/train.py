from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from sklearn.metrics import roc_curve, auc
from operator import itemgetter

class Get_loss(nn.Module):
    def __init__(self): # lambda_reg
        super(Get_loss, self).__init__()
        # self.lambda_reg = lambda_reg  # 正则化参数

    def forward(self, one_index, zero_index, target, input, dd, mm, opt): 
        """
        计算新的损失：包括预测误差和正则化项
        :param one_index: 真实连接的节点对
        :param zero_index: 非连接的节点对
        :param target: 真实标签（评分）
        :param input: 模型预测值
        :param dd, mm: 额外的输入数据（可选）
        :param opt: 配置项
        :return: 计算的损失值
        """
        # 计算MSE损失
        mse_loss = nn.MSELoss(reduction='none')
        loss_matrix = mse_loss(input, target)

        # 计算预测误差
        loss = (1 - opt.positive_weight) * loss_matrix[one_index].sum() + opt.positive_weight * loss_matrix[zero_index].sum()

        # 计算正则化项 (L2范数正则化)
        # regularization_loss = self.lambda_reg * (torch.norm(dd, 'fro')**2 + torch.norm(mm, 'fro')**2) / 2

        # 总损失
        total_loss = loss #+ regularization_loss
        return total_loss
    
def train_epoch(model, train_data, optimizer, regression_crit, one_index, zero_index, opt, epoch_losses):  
    """
    训练一个epoch
    """
    model.zero_grad()  # 清零梯度
    score = model(train_data)
    loss = regression_crit(one_index, zero_index, train_data[4].cuda(), score, train_data[0], train_data[1],opt)

    # 记录损失
    epoch_losses.append(loss.item())
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    return loss


def train_model(model, train_data, optimizer, opt):  
    """
    训练整个模型
    """
    model.train()
    regression_crit = Get_loss()
    one_index = train_data[2]['one'].cuda().t().tolist()  # 有联系的节点
    zero_index = train_data[2]['zero'].cuda().t().tolist()
    
    epoch_losses = []  # 存储每个epoch的损失
    for epoch in range(1, opt.epoch + 1):
        loss = train_epoch(model, train_data, optimizer, regression_crit, one_index, zero_index, opt, epoch_losses)
      
    # plt.plot(range(1, opt.epoch + 1), epoch_losses, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss over Epochs')
    # plt.legend()
    # plt.show()

def test_model(model, train_data, X_best, Y_best, best_roc_auc):
    """
    测试整个模型并计算ROC AUC，同时更新全局最佳ROC AUC和ROC曲线
    """
    model.eval()
    score = model(train_data).tolist()
    test_index = train_data[2]['test']
    zero_index = train_data[2]['zero']

    check = []
    for i in range(test_index.size(0)):
        u = test_index[i][0]
        v = test_index[i][1]
        check.append([score[u][v], 1])
    for i in range(zero_index.size(0)):
        u = zero_index[i][0]
        v = zero_index[i][1]
        check.append([score[u][v], 0])

    X = [0]
    Y = [0]
    check = sorted(check, key=itemgetter(0), reverse=True)
    P, N = 0, 0

    for i in range(zero_index.size(0) + test_index.size(0)):
        if check[i][1] == 1:
            P += 1
        else:
            N += 1
        TP = P
        FP = i + 1 - P
        TN = zero_index.size(0) - N
        FN = test_index.size(0) - P
        X.append(FP / (TN + FP))  # False Positive Rate
        Y.append(TP / (TP + FN))  # True Positive Rate

    roc_auc = auc(X, Y)

    # 如果当前AUC超过全局最佳AUC，更新全局最佳AUC及对应的X, Y
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        X_best, Y_best = X, Y  # 更新全局最佳ROC曲线

    return roc_auc, X_best, Y_best, best_roc_auc



def adjust_learning_rate(optimizer, roc_auc, previous_roc_auc, patience, patience_counter,no_improvement_count,best_roc_auc):
    """
    根据ROC AUC值调整学习率，引入耐心机制
    - patience_counter: 记录已经连续多少轮ROC AUC没有提升
    - previous_roc_auc: 记录上一次的ROC AUC值
    - patience: 当ROC AUC没有提升多少轮时，才减少学习率
    """
    if  roc_auc >= best_roc_auc :    
        patience_counter = 0.0  # 重置耐心计数器
    # if roc_auc >= previous_roc_auc and roc_auc < best_roc_auc :  # 如果AUC提升
    #     patience_counter += 0.5 # 
    else:  # 如果AUC没有提升
        patience_counter += 1.0  # 增加耐心计数器
    print('1',patience_counter)
    print('2',best_roc_auc)   
    print('3',previous_roc_auc)
    print('4',roc_auc)
    # 当耐心计数器达到设定的阈值时，减小学习率
    if patience_counter > patience:
        # 学习率调整策略：逐步减小学习率
        if optimizer.param_groups[0]['lr'] > 0.00001:
            new_lr = optimizer.param_groups[0]['lr'] * 0.7  # 降低学习率
            optimizer.param_groups[0]['lr'] = new_lr
            print(f"Learning rate reduced to {new_lr:.6f}")
        patience_counter = 0  # 重置耐心计数器


    return  patience_counter ,no_improvement_count

