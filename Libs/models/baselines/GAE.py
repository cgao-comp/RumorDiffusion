"""Graph Autoencoder (GAE) for motivation experiment.

This module implements a Graph Autoencoder using PyTorch Geometric's GAE
wrapper with a GCN-based encoder for learning graph representations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GAE, GCNConv


class GCNEncoder(nn.Module):
    """GCN-based encoder for Graph Autoencoder.
    
    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden units in first layer.
        out_channels: Dimension of output embeddings.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels].
            edge_index: Edge indices of shape [2, num_edges].
            
        Returns:
            Node embeddings of shape [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def create_gae_model(in_channels: int, 
                     hidden_channels: int = 64, 
                     out_channels: int = 128) -> GAE:
    """Creates a GAE model with GCN encoder.
    
    Args:
        in_channels: Number of input node features.
        hidden_channels: Number of hidden units.
        out_channels: Dimension of output embeddings.
        
    Returns:
        Initialized GAE model.
    """
    encoder = GCNEncoder(in_channels, hidden_channels, out_channels)
    model = GAE(encoder)
    return model

