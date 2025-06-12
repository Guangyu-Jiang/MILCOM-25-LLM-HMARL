import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128):
        super(GNNPolicy, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self.node_policy = nn.Linear(embedding_dim, 3)  # Node-level actions
        self.edge_policy = nn.Linear(embedding_dim * 2, 2)  # Edge-level actions
        self.global_policy = nn.Linear(embedding_dim, 1)  # Global actions
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        if batch is not None:
            g = global_mean_pool(x, batch)
        else:
            g = x.mean(dim=0, keepdim=True)
            
        # Value head
        v = F.relu(self.fc1(g))
        value = self.fc2(v)
        
        # Policy heads
        node_logits = self.node_policy(x)  # Node actions
        
        # Edge actions - combine node features for each edge
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_logits = self.edge_policy(edge_features)
        
        # Global actions
        global_logits = self.global_policy(g)
        
        return node_logits, edge_logits, global_logits, value 