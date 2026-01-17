import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, out_dim, dropout=0.5):
        super(GCN, self).__init__()
     
        self.conv1 = GCNConv(num_features, hidden_dim)
  
        self.conv2 = GCNConv(hidden_dim, out_dim) 
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

