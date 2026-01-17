import torch.nn as nn
import numpy as np
from utils import *
from DilatedAttention import *
from EdgeConditioned import *
from GCATransformer import *
from Transfor import *
from GCN import *
from GAT import *

class MLP(nn.Module):
    def __init__(self, embedding_size, drop_rate):
        super(MLP, self).__init__()
        self.embedding_size = embedding_size
        self.drop_rate = drop_rate

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.mlp_prediction = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 2, self.embedding_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 4, self.embedding_size // 6),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.embedding_size // 6, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_prediction.apply(init_weights)

    def forward(self, rd_features_embedding):
        predict_result = self.mlp_prediction(rd_features_embedding)
        return predict_result
    
class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)# 定义Chebyshev卷积层，K表示多项式阶数，能捕获多阶邻居
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class hu(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.in_dim
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.output_dim = args.output_dim#1024
        self.hidden_dim = args.hidden_dim
        self.Dilate_heads = args.Dilate_heads
        fout_dim = args.fout_dim
        self.Sa = args.Sa
        self.GNN_layer= args.GNN_layer
        self.mlp_prediction = MLP(1664, 0.2) #MDAD
        #self.mlp_prediction = MLP(640, 0.2) #DrugVirus
        #self.mlp_prediction = MLP(2176, 0.2) #aBiofilm
        self.hu = DilatedAttention(1024, self.hidden_dim,self.Dilate_heads,  window_size=7, bidirectional=True)#256
        self.layers = nn.ModuleList([GraphTransformerLayer(self.input_dim, self.hidden_dim, fout_dim, 0.2,
                     layer_norm=False, batch_norm=True, residual=True, use_bias=False) for _ in range(self.Sa- 1)])#256,128,64,8,0.4,true,false,true,9
        self.layers.append(
            GraphTransformerLayer(self.input_dim, self.hidden_dim, fout_dim, 0.2, False, True,
                                  True))
      
        self.Kk = GNNEncoder(1546, 1024, 1, self.GNN_layer)#MDAD
        #self.Kk = GNNEncoder(270, 256, 1, self.GNN_layer)#DrugVirus
        #self.Kk = GNNEncoder(1860, 1024, 1, self.GNN_layer)#aBiofilm
        self.Gnn = GCN(1546, 1024,1024)   
        self.Gat = GAT(1024,512,512)


        self.FN = nn.Linear(1546, 1024)  #MDAD
        #self.FN = nn.Linear(270, 256)  #DrugVirus
        #self.FN = nn.Linear(1860, 1024)  #aBiofilm
        
        self.FNN = nn.Linear(128, 64) 
    def forward(self, args, x, rel_matrix, A1, edge_index, train_model):
        out1 = self.Kk(x,edge_index,A1)
        x1 = self.FN(x)
        out2 = self.hu(x1)
        
        for conv in self.layers:
            h = conv(x)
        features_embedding = [out1,out2,h]  
        features_embedding = torch.cat(features_embedding, dim=1)
        outputs = F.leaky_relu(features_embedding)
        if train_model:
            train_features_inputs, train_lable = train_features_choose(rel_matrix, outputs)
            train_mlp_result = self.mlp_prediction(train_features_inputs)
            return train_mlp_result, train_lable
        else:
            test_features_inputs, test_lable = test_features_choose(rel_matrix, outputs)
            test_mlp_result = self.mlp_prediction(test_features_inputs)
            return test_mlp_result, test_lable, outputs
       

