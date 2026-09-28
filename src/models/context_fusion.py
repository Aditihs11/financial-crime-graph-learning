import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear

class AdaptiveContextFusionNet(nn.Module):
    """
    Adaptive Multi-Entity Context Fusion (AMECF) model.
    Uses Heterogeneous Graph Transformer (HGT) architecture to adaptively learn 
    contextual attention weights across all 14 node types and 18 edge relations.
    """
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata, node_features_dict):
        super().__init__()
        
        self.lin_dict = nn.ModuleDict()
        for node_type, num_features in node_features_dict.items():
            self.lin_dict[node_type] = Linear(num_features, hidden_channels)
            
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)

            self.convs.append(conv)
            
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x_dict, edge_index_dict):
        # 1. Type-specific projection to a shared hidden space
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = F.gelu(self.lin_dict[node_type](x))
            
        # 2. Adaptive Context Fusion via Heterogeneous Attention Layers
        for conv in self.convs:
            x_dict_proj = conv(x_dict_proj, edge_index_dict)
            
        # 3. Predict over 'account' nodes
        account_embed = x_dict_proj['account']
        out = self.classifier(account_embed)
        
        return out
