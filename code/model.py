import torch as t
from torch import nn
from torch_geometric.nn import conv , GATConv



# 模型Ⅱ
class Model(nn.Module):
    def __init__(self, sizes):
        super(Model, self).__init__()

        self.m = sizes.m
        self.d = sizes.d
        
        self.fg = 256          
        self.fd = 256
        self.k = 64
        self.a =1
        self.b =0
        self.c =0

        # self.beta1 = t.nn.Parameter(t.FloatTensor([1/2]))
        # self.beta2 = t.nn.Parameter(t.FloatTensor([1/2]))
        # self.beta1 = t.nn.Parameter(t.FloatTensor([1/3,1/3]))
        # self.beta2 = t.nn.Parameter(t.FloatTensor([1/3,1/3]))
        self.beta1 = nn.Parameter(t.FloatTensor([1/3, 1/3]))  
        self.beta2 = nn.Parameter(t.FloatTensor([1/3, 1/3])) 

        # 添加GATConv（图注意力层）以增强信息聚合#     多个注意力头的输出拼接（concat=True）生成的输出维度是 64 * 4 = 256。
        self.gat_x1 = GATConv(self.fg, 64, heads=4, concat=True)
        self.gat_y1 = GATConv(self.fd, 64, heads=4, concat=True)
        # self.gat_x1 = GATConv(self.fg, 256, heads=4, concat=False)
        # self.gat_y1 = GATConv(self.fd, 256, heads=4, concat=False)   
        # 计算时间增长且为未提升
        self.gcn_x1 = conv.GCNConv(self.fg, self.fg)    # 两个参数，前面是in_channel，后面是out_channel，也即GCN网络的每个节点的向量长度
        self.gcn_y1 = conv.GCNConv(self.fd, self.fd)
        self.gcn_x2 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y2 = conv.GCNConv(self.fd, self.fd)
        self.gcn_x3 = conv.GCNConv(self.fg, self.fg)
        self.gcn_y3 = conv.GCNConv(self.fd, self.fd)

        # self.gcn_x4 = conv.GCNConv(self.fg, self.fg)    # 两个参数，前面是in_channel，后面是out_channel，也即GCN网络的每个节点的向量长度
        # self.gcn_y4 = conv.GCNConv(self.fd, self.fd)
        # self.gcn_x5 = conv.GCNConv(self.fg, 128)
        # self.gcn_y5 = conv.GCNConv(self.fd, 128)
        # self.gcn_x6 = conv.GCNConv(128, 64)
        # self.gcn_y6 = conv.GCNConv(128, 64)



        self.linear_x_1 = nn.Linear(self.fg, 256)       # 两个参数，相当于前面是in_channel，后面是out_channel
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(self.fd, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
 

    def forward(self, input):
        t.manual_seed(1)  # 为CPU设置种子

        # 随机生成miRNA和疾病的特征矩阵
        x_m = t.randn(self.m, self.fg).cuda()  # miRNA特征矩阵
        x_d = t.randn(self.d, self.fd).cuda()  # 疾病特征矩阵
        beta1, beta2 = self.beta1, self.beta2  # 获取beta1和beta2

        # GCN + GAT 组合特征学习
        X1 = self.leaky_relu(self.gcn_x1(x_m, input[1]['edge_index'].cuda(), input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
        Y1 = self.leaky_relu(self.gcn_y1(x_d, input[0]['edge_index'].cuda(), input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))

        # 使用GATConv增强图卷积学习
        X1 = self.leaky_relu(self.gat_x1(X1, input[1]['edge_index'].cuda()))
        Y1 = self.leaky_relu(self.gat_y1(Y1, input[0]['edge_index'].cuda()))

        # GCN层1
        X2 = self.leaky_relu(self.gcn_x2(X1, input[1]['edge_index'].cuda(), input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
        Y2 = self.leaky_relu(self.gcn_y2(Y1, input[0]['edge_index'].cuda(), input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))

        # GCN层2
        X3 = self.leaky_relu(self.gcn_x3(X2, input[1]['edge_index'].cuda(), input[1]['data'][input[1]['edge_index'][0], input[1]['edge_index'][1]].cuda()))
        Y3 = self.leaky_relu(self.gcn_y3(Y2, input[0]['edge_index'].cuda(), input[0]['data'][input[0]['edge_index'][0], input[0]['edge_index'][1]].cuda()))

        # 线性组合三个层的输出
        X = beta1[0]*X1 + beta1[1]*X2 + (1-beta1[0]-beta1[1])*X3
        Y = beta2[0]*Y1 + beta2[1]*Y2 + (1-beta2[0]-beta2[1])*Y3

        # 全连接层（分别对miRNA和疾病特征进行处理）
        x1 = self.leaky_relu(self.linear_x_1(X))
        x2 = self.leaky_relu(self.linear_x_2(x1))
        x = self.leaky_relu(self.linear_x_3(x2))

        y1 = self.leaky_relu(self.linear_y_1(Y))
        y2 = self.leaky_relu(self.linear_y_2(y1))
        y = self.leaky_relu(self.linear_y_3(y2))

        # 输出miRNA与疾病之间的关系（通过点积）
        return x.mm(y.t())
