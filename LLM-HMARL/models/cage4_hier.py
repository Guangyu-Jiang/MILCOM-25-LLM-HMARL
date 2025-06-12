import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from .cage4 import GNNPolicy

class GNNHierPolicy(GNNPolicy):
    def __init__(self, input_dim, num_skills=4, hidden_dim=256, embedding_dim=128):
        super(GNNHierPolicy, self).__init__(input_dim, hidden_dim, embedding_dim)
        self.num_skills = num_skills
        
        # Skill selection head
        self.skill_selector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills)
        )
        
    def forward(self, x, edge_index, batch=None, skill_id=None):
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
        
        # Skill selection
        skill_logits = self.skill_selector(g)
        
        if skill_id is not None:
            # If skill is provided, return action logits for that skill
            node_logits = self.node_policy(x)
            edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
            edge_logits = self.edge_policy(edge_features)
            global_logits = self.global_policy(g)
            return node_logits, edge_logits, global_logits, value
        else:
            # Otherwise just return skill selection logits and value
            return skill_logits, value 