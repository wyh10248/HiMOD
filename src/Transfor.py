import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import VGAE
import numpy as np
from utils import *
import argparse
device = torch.device("cpu")


class MultiheadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):#256,128,8
        super(MultiheadAttention, self).__init__()
        assert in_dim % num_heads == 0
        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.num_heads = num_heads
        self.depth = in_dim // num_heads#128
        self.out_dim = out_dim
        self.query_linear = nn.Linear(in_dim, in_dim)
        self.key_linear = nn.Linear(in_dim, in_dim)
        self.value_linear = nn.Linear(in_dim, in_dim)
        self.output_linear = nn.Linear(in_dim, out_dim)


    def split_heads(self, x, batch_size):
        # reshape input to [batch_size, num_heads, seq_len, depth]

        x_szie = x.size()[:-1] + (self.num_heads, self.depth)#511*8*32
        x = x.reshape(x_szie)
        return x.transpose(-1, -2)#将最后两个维度交换511*32*8

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)#652

        # Linear projections
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Split the inputs into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))#所以scores的shape最终是(1024, 128, 128),它对Q和K的乘法做了规范化处理,使得点积在0-1范围内。

        # Apply mask (if necessary)
        if mask is not None:
            mask = mask.unsqueeze(1)  # add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=0)
        attention_output = torch.matmul(attention_weights, V)

        # Merge the heads
        output_size = attention_output.size()[:-2]+ (query.size(1),)
        attention_output = attention_output.transpose(-1, -2).reshape((output_size))

        # Linear projection to get the final output
        attention_output = self.output_linear(attention_output)#1024*256

        return torch.sigmoid(attention_output)


class GraphTransformerLayer1(nn.Module):
    """
        Param:
    """
 
    def __init__(self, in_dim, hidden_dim, fout_dim, num_heads, dropout, layer_norm=False, batch_norm=True, residual=True,
                 use_bias=False):#256,128,64,8,0.4,true,false,true,9
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.fout_dim = fout_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiheadAttention(in_dim, hidden_dim, num_heads)#256,128,8

        self.residual_layer1 = nn.Linear(in_dim, fout_dim)  #残差256,64

        self.O = nn.Linear(hidden_dim, fout_dim)#128*64

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(fout_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(fout_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(fout_dim, fout_dim * 2)
        self.FFN_layer2 = nn.Linear(fout_dim * 2, fout_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(fout_dim)#64

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(fout_dim)

    def forward(self, h):
        h_in1 = self.residual_layer1(h)  # for first residual connection
        # multi-head attention out
        attn_out = self.attention(h, h, h)#h=652*652,attn_out=#1024*256
        #h = attn_out.view(-1, self.out_channels)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = F.leaky_relu(self.O(attn_out))#128*64

        if self.residual:
            attn_out = h_in1 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm1(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm1(attn_out)

        h_in2 = attn_out  # for second residual connection

        # FFN
        attn_out = self.FFN_layer1(attn_out)
        attn_out = F.leaky_relu(attn_out)
        attn_out = F.dropout(attn_out, self.dropout, training=self.training)
        attn_out = self.FFN_layer2(attn_out)
        attn_out = F.leaky_relu(attn_out)

        if self.residual:
            attn_out = h_in2 + attn_out  # residual connection

        if self.layer_norm:
            attn_out = self.layer_norm2(attn_out)

        if self.batch_norm:
            attn_out = self.batch_norm2(attn_out)

        return attn_out

class GTM_net(nn.Module):
    def __init__(self, args, X):
        super().__init__()
        self.X = X
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        in_dim = 256
        out_dim = 256
        fout_dim = 128
        head_num = 4
        dropout = 0.2
        self.Sa = 8
        self.output = 256
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.FNN = nn.Linear(1546, 256) 

        self.layers = nn.ModuleList([GraphTransformerLayer1(in_dim, out_dim, fout_dim, head_num,dropout,
                                                            self.layer_norm, self.batch_norm, self.residual, args) for _ in range(self.Sa- 1)])#256,128,64,8,0.4,true,false,true,9
        self.layers.append(
            GraphTransformerLayer1(in_dim, out_dim, fout_dim, head_num, dropout, self.layer_norm, self.batch_norm,
                                  self.residual))
        self.FN = nn.Linear(fout_dim, self.output)
        
    def forward(self, x, edge_index, rel_matrix):
        x1 = self.FNN(x)
        X = F.leaky_relu(x1)
        #print("X shape:", X.shape)
        for conv in self.layers:
            h = conv(X) 
        return h

       
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float,help='learning rate')
    parser.add_argument('--batch_size',default=32,type=int,help="train/test batch size")
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--k_fold', type=int, default=5, help='crossval_number.')
    parser.add_argument('--epoch', type=int, default=1, help='train_number.')
    parser.add_argument('--in_dim', type=int, default=256, help='in_feature.')
    parser.add_argument('--out_dim', type=int, default=128, help='out_feature.')
    parser.add_argument('--fout_dim', type=int, default=64, help='f-out_feature.')
    parser.add_argument('--output_t', type=int, default=64, help='finally_out_feature.')
    parser.add_argument('--head_num', type=int, default=8, help='head_number.')
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout.')
    parser.add_argument('--pos_enc_dim', type=int, default=64, help='pos_enc_dim.')
    parser.add_argument('--residual', type=bool, default=True, help='RESIDUAL.')
    parser.add_argument('--layer_norm', type=bool, default=True, help='LAYER_NORM.')
    parser.add_argument('--batch_norm', type=bool, default=False, help='batch_norm.')
    parser.add_argument('--Sa', type=int, default=8, help='TransformerLayer.') 
    args = parser.parse_args()

    data_path = '../dataset/'
    data_set = 'data1/'

    A = np.loadtxt(data_path + data_set + 'disease-lncRNA.csv',delimiter=',')
    A=A.T
    disSimi1 = loadmat(data_path + data_set + 'diease_similarity_kernel-k=40-t=20-ALPHA=0.1.mat') 
    disSimi = disSimi1['WM']
    RNASimi1 = loadmat(data_path + data_set + 'RNA_s_kernel-k=40-t=20-ALPHA=0.1.mat')
    lncSimi = RNASimi1['WM']
    x=constructHNet(A, lncSimi, disSimi)
    edge_index=adjacency_matrix_to_edge_index(A)
    train_matrix = torch.from_numpy(A).to(torch.float32)
    x = x.to(torch.float32)
    edge_index = edge_index.to(torch.int64)
    model = GTM_net(args, x).to(device)
    S = model(x, edge_index, train_matrix)
    print(S.shape)


