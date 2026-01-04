import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import *


# ===== Global Context Attention =====
class GlobalContextAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GlobalContextAttention, self).__init__()
        self.query_linear = nn.Linear(in_dim, out_dim)
        self.key_linear = nn.Linear(in_dim, out_dim)
        self.value_linear = nn.Linear(in_dim, out_dim)
        self.output_linear = nn.Linear(out_dim, out_dim)

    def forward(self, h):
        """
        h: [batch_size, num_nodes, in_dim]
        """
        # 1. 计算全局上下文向量
        global_context = torch.mean(h, dim=1, keepdim=True)  # [B, 1, in_dim]

        # 2. 映射到 Key 空间
        global_key = self.key_linear(global_context)  # [B, 1, out_dim]

        # 3. 节点映射到 Query 空间
        Q = self.query_linear(h)  # [B, N, out_dim]

        # 4. 节点与全局上下文的相似度（注意力权重）
        attn_scores = torch.sigmoid(torch.matmul(Q, global_key.transpose(-1, -2)))  # [B, N, 1]

        # 5. 将节点特征映射到 Value 空间
        V = self.value_linear(h)  # [B, N, out_dim]

        # 6. 注意力加权更新
        updated_h = attn_scores * V  # [B, N, out_dim]

        # 7. 输出变换
        output = self.output_linear(updated_h)  # [B, N, out_dim]

        return output


# ===== Graph Transformer Layer with GCA =====
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, fout_dim, dropout,
                 layer_norm=False, batch_norm=True, residual=True, use_bias=False):
       
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fout_dim = fout_dim
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # 替换为 Global Context Attention
        self.attention = GlobalContextAttention(in_dim, hidden_dim)

        # 残差映射
        self.residual_layer1 = nn.Linear(in_dim, fout_dim)

        # 将注意力输出映射到 fout_dim
        self.O = nn.Linear(hidden_dim, fout_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(fout_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(fout_dim)

        # 前馈网络 FFN
        self.FFN_layer1 = nn.Linear(fout_dim, fout_dim * 2)
        self.FFN_layer2 = nn.Linear(fout_dim * 2, fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(fout_dim)
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(fout_dim)

    def forward(self, h):
        """
        h: [batch_size, num_nodes, in_dim]
        """
        if h.dim() == 2:  # [N, in_dim]
           h = h.unsqueeze(0)  # 变成 [1, N, in_dim]
        h_in1 = self.residual_layer1(h)  # 残差分支1

        # Global Context Attention
        attn_out = self.attention(h)  # [B, N, hidden_dim]
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))  # [B, N, fout_dim]

        # 残差连接
        if self.residual:
            attn_out = h_in1 + attn_out
        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)
        if self.batch_norm:
            B, N, C = attn_out.shape
            attn_out = self.batch_norm1(attn_out.view(B * N, C)).view(B, N, C)

        h_in2 = attn_out  # 残差分支2

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)

        # 残差连接
        if self.residual:
            attn_out = h_in2 + attn_out
        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)
        if self.batch_norm:
            B, N, C = attn_out.shape
            attn_out = self.batch_norm2(attn_out.view(B * N, C)).view(B, N, C)

        return attn_out.squeeze(0) 
if __name__== '__main__':
    dataset = 'MDAD' 
    #dataset = 'DrugVirus'   
    #dataset = 'aBiofilm'  #microbe 140  drug 1720
    if dataset == 'MDAD':
        embed_dim_default = 1546
        hidden_dim = 512
        output_dim = 256
        epoch = 15
    elif dataset == 'DrugVirus':
        embed_dim_default = 270
        hidden_dim = 256
        output_dim = 128
        epoch = 25
    elif dataset == 'aBiofilm':
        embed_dim_default = 1860
        hidden_dim = 1024
        output_dim = 256
        epoch = 20
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
    parser.add_argument('--epoch', type=int, default=epoch, help='train_number.')
    parser.add_argument('--in_dim', type=int, default=embed_dim_default, help='in_feature.')
    parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help='hidden_dim.')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim.')
    parser.add_argument('--num_layers', type=int, default=3, help='num_layers.')
    parser.add_argument('--embed_dim', type=int, default=embed_dim_default, help='embed_dim.')
    parser.add_argument('--num_heads', type=int, default=8, help='head_number.')
    parser.add_argument('--output_dim', type=int, default=output_dim, help='output_dim.')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='drop_rate.')
    parser.add_argument('--K', type=int, default=4, help='_multi_order_diffusion.')
    parser.add_argument('--n_layer', type=int, default=4, help='Mamba_layer')
    parser.add_argument('--GT_heads', type=int, default=8, help='GT_heads')
    parser.add_argument('--fout_dim', type=int, default=128, help='fout_dim')
    parser.add_argument('--Sa', type=int, default=4, help='GT_layer')

    args = parser.parse_args()

    data_path = '../dataset/'
    data_set = 'MDAD/'

    A = np.loadtxt(data_path + data_set + 'drug_microbe_adjacency.csv',delimiter=',')
    DSM = np.loadtxt(data_path + data_set + 'DSM1.csv',delimiter=',') 
    MSM = np.loadtxt(data_path + data_set + 'MSM1.csv',delimiter=',')
    x=constructHNet(A, DSM, MSM)
    edge_index,_ =adjacency_matrix_to_edge_index(A)
    train_matrix = torch.from_numpy(A).to(torch.float32)
    x = torch.from_numpy(x).to(torch.float32)
    edge_index,_ = edge_index.to(torch.int64)
    model = GraphTransformerLayer(args.in_dim, args.hidden_dim, args.fout_dim, args.GT_heads, 0.2, False, True,
                          True)
    S = model(x)
    print(S.shape)




