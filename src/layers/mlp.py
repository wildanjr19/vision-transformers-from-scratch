"""
Multi Layer Perceptron
"""
import torch
import torch.nn as nn

from config.CONFIG import EMBED_DIM, HIDDEN_DIM

class MLP(nn.Module):
    """
    Input embed_dim dan dikembalikan ke embed_dim
    Args:
        - in_features: dimensi input (embed_dim)
        - hidden_features: dimensi hidden layer (embed_dim * mlp_ratio) -> CONFIG
        - out_featueres: dimensi output (embed_dim)
        - dropout
    """
    def __init__(self, in_features: int = EMBED_DIM, hidden_features: int = HIDDEN_DIM, out_features: int = EMBED_DIM, drop: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x