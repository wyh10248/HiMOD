import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, heads=8, dropout=0.01):
        super().__init__()
      
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim // heads, heads=heads, dropout=dropout)
       
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, dropout=dropout)
      
        self.gat3 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim // heads, heads=heads, dropout=dropout)
        
        self.gat4 = GATConv(in_channels=hidden_dim, out_channels=out_dim // heads, heads=heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat3(x, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat4(x, edge_index)  
        return x