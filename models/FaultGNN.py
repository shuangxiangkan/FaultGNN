import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FaultGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, heads=4):
        """
        初始化 FaultGNN 模型（现基于GraphSAGE）。

        Args:
            input_dim (int): 输入维度。
            hidden_dim (int): 隐藏层维度。
            num_layers (int): GraphSAGE 层数。
            heads (int): 保留参数以兼容接口（GraphSAGE不使用heads）。
        """
        super(FaultGNN, self).__init__()
        self.num_layers = num_layers
        
        # 添加特征变换层，增强输入特征的表现力
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层（使用变换后的特征维度）
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层
        self.convs.append(SAGEConv(hidden_dim, 2))
        
        self.dropout = 0.3

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播。

        Args:
            x (Tensor): 节点特征矩阵。
            edge_index (Tensor): 边索引。
            edge_attr (Tensor, optional): 边特征。GraphSAGE不使用边特征，保留以兼容接口。

        Returns:
            Tensor: 分类 logits。
        """
        # 先对输入特征进行变换，增强表现力
        x = self.feature_transform(x)
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
