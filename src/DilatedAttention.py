import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DilatedAttention(nn.Module):
    """
    稀释注意力机制：在固定大小窗口内以膨胀间隔（dilation=2）采样邻居
    """
    
    def __init__(self, input_dim, dim=512, num_heads=8, window_size=7, 
                 bidirectional=True, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert window_size % 2 == 1, "window_size should be odd"
        
        self.input_dim = input_dim
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.dilation = 8
        self.bidirectional = bidirectional
        self.scale = self.head_dim ** -0.5

        # 输入线性变换：防止输入维度 input_dim 不可整除
        self.input_proj = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        
        self.tse = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def create_dilated_pattern(self, seq_len, device):
        neighbors = []
        half_window = self.window_size // 2
        for i in range(seq_len):
            pos_neighbors = [i]
            for step in range(1, half_window + 1):
                left_idx = i - self.dilation * step
                if left_idx >= 0:
                    pos_neighbors.append(left_idx)
            if self.bidirectional:
                for step in range(1, half_window + 1):
                    right_idx = i + self.dilation * step
                    if right_idx < seq_len:
                        pos_neighbors.append(right_idx)
            neighbors.append(sorted(pos_neighbors))
        return neighbors
    
    def create_sparse_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        neighbors = self.create_dilated_pattern(seq_len, device)
        for i, neighbor_list in enumerate(neighbors):
            for j in neighbor_list:
                mask[i, j] = 0.0
        return mask, neighbors
    
    def forward(self, x, return_attention=False):
        """
        x: [seq_len, input_dim]
        """
        seq_len, _ = x.size()

        # 投影到dim维度
        x = self.input_proj(x)  # [seq_len, dim]

        # tse生成
        tse = self.tse(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        tse = tse.permute(1, 2, 0, 3)  # [3, H, L, D]
        target, support, evidence = tse[0], tse[1], tse[2]
        
        # 注意力分数
        attn_scores = torch.matmul(target, support.transpose(-2, -1)) * self.scale
        
        # 稀疏掩码
        sparse_mask, neighbors = self.create_sparse_mask(seq_len, x.device)
        attn_scores = attn_scores + sparse_mask.unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, evidence)
        out = out.transpose(1, 2).contiguous().view(seq_len, self.dim)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights, neighbors
        return out

# 使用示例
if __name__ == "__main__":
    model = DilatedAttention(
        input_dim=500,   # 输入维度可以不是 dim
        dim=500,         # 内部会映射为 500
        num_heads=10, 
        window_size=7,
        bidirectional=True
    )
    
    x = torch.randn(128, 500)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    
# if __name__== '__main__':
#     # 数据加载和预处理
#     matrix1 = np.loadtxt('D:/Desktop文件/MDA/dataset/MDAD/DSM.csv', delimiter=',')
#     matrix2 = np.loadtxt('D:/Desktop文件/MDA/dataset/MDAD/MSM.csv', delimiter=',')
#     A = np.loadtxt('D:/Desktop文件/MDA/dataset/MDAD/drug_microbe_adjacency.csv', delimiter=',') 
#     X = constructHNet(A, matrix1, matrix2)  
#     X  = torch.from_numpy(X).to(torch.float32)
#     model2 = EfficientDilatedSelfAttention(1546, dilation=2, window_size=3, bidirectional=True)
#     out2 = model2(X)
#     print(f"高效版本输出形状: {out2.shape}")