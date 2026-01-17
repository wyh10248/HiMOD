import torch
import torch.nn as nn
import torch.nn.functional as F

def make_sparse_adj(edge_index, edge_weight, num_nodes, symmetric=False, device=None):
    device = device if device is not None else edge_index.device
    if symmetric:
        src = edge_index[0]
        dst = edge_index[1]
        src2 = torch.cat([src, dst], dim=0)
        dst2 = torch.cat([dst, src], dim=0)
        idx = torch.stack([src2, dst2], dim=0)
        vals = torch.cat([edge_weight, edge_weight], dim=0)
    else:
        idx = edge_index
        vals = edge_weight
    sparse = torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device)
    sparse = sparse.coalesce()
    return sparse

class EdgeConditionedLayerV2(nn.Module):
    def __init__(self, in_dim, out_dim, edge_feat_dim,
                 use_2hop=True, gate_mode='scalar', normalize=True,
                 dropedge_rate=0.0, symmetric_adj=True):
        super().__init__()
        assert gate_mode in ('scalar','vector'), "gate_mode must be 'scalar' or 'vector'"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_feat_dim = edge_feat_dim
        self.use_2hop = use_2hop
        self.gate_mode = gate_mode
        self.normalize = normalize
        self.dropedge_rate = float(dropedge_rate)
        self.symmetric_adj = symmetric_adj
        
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.lin1 = nn.Linear(in_dim, out_dim, bias=False)
        self.lin2 = nn.Linear(in_dim, out_dim, bias=False)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, max(edge_feat_dim, 32)),
            nn.ReLU(),
            nn.Linear(max(edge_feat_dim, 32), 1)
        )
        
        if gate_mode == 'scalar':
            self.g = nn.Parameter(torch.tensor(0.5))
        else:
            self.g = nn.Parameter(torch.ones(out_dim) * 0.5)
        
        self.update = nn.Linear(out_dim + in_dim, out_dim)
        self.eps = 1e-12

    def forward(self, H, edge_index, edge_attr):

        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.from_numpy(edge_attr).to(H.device)
        else:
            edge_attr = edge_attr.to(H.device)

        device = H.device
        N = H.size(0)

        edge_logits = self.edge_mlp(edge_attr).squeeze(-1)
        edge_w = torch.sigmoid(edge_logits)

        if self.training and self.dropedge_rate > 0.0:
            mask = (torch.rand_like(edge_w) >= self.dropedge_rate)
            edge_w = edge_w[mask]
            edge_index = edge_index[:, mask]

        sparse_A = make_sparse_adj(edge_index, edge_w, num_nodes=N, symmetric=self.symmetric_adj, device=device)

        if self.normalize:
            vals = sparse_A.values()
            idx = sparse_A.indices()
            rows = idx[0]
            deg = torch.zeros(N, device=device).index_add(0, rows, vals)
            deg_inv_sqrt = torch.pow(deg + self.eps, -0.5)
            cols = idx[1]
            vals_norm = vals * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]
            sparse_A = torch.sparse_coo_tensor(idx, vals_norm, (N, N), device=device).coalesce()

        h_self = self.self_lin(H)
        h1_src_proj = self.lin1(H)
        h1 = torch.sparse.mm(sparse_A, h1_src_proj)

        if self.use_2hop:
            h2_src_proj = self.lin2(H)
            tmp = torch.sparse.mm(sparse_A, h2_src_proj)
            h2 = torch.sparse.mm(sparse_A, tmp)
        else:
            h2 = torch.zeros_like(h1)

        if self.gate_mode == 'scalar':
            g = torch.clamp(self.g, 0.0, 1.0)
            agg = g * h1 + (1.0 - g) * h2
        else:
            g_vec = torch.sigmoid(self.g)
            agg = h1 * g_vec.view(1, -1) + h2 * (1.0 - g_vec.view(1, -1))

        combined = torch.cat([agg, H], dim=-1)
        out = self.update(combined)
        return F.relu(out)

    
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_feat_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EdgeConditionedLayerV2(in_dim, hidden_dim, edge_feat_dim))
        for _ in range(num_layers - 1):
            self.layers.append(EdgeConditionedLayerV2(hidden_dim, hidden_dim, edge_feat_dim))
    
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x
    
    