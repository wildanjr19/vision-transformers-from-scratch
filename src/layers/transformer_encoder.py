"""
Block transformer encoder
layer norm, residual connection, mha, mlp
# phase 1
    - Layer Norm + MHA + residual connection
# phase 2
    - layer Norm + MLP + residual connection
"""

import torch 
import torch.nn as nn

from config.CONFIG import EMBED_DIM, ATTENTION_HEADS, HIDDEN_DIM
from src.layers.mlp import MLP
from src.layers.attention import Attention

class TransformerEncoderBlock(nn.Module):
    """
    Args:
        - embed_dim: dimensi embedding input dan output final
        - num_heads : untuk mha
        - hidden_dim : dimensi hidden layer di mlp
        - qkv_bias : apakah menambahkan bias di qkv linear layer
        - drop : dropout rate
        - attn_drop : attention dropout rate
    """
    def __init__(self, embed_dim = EMBED_DIM, num_heads = ATTENTION_HEADS, hidden_dim = HIDDEN_DIM, qkv_bias = True, drop = 0, attn_drop = 0):
        super().__init__()

        # layernorm pertama (sebelum attn)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        # multi head attention
        self.attention = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        # layernorm kedua (sebelum mlp)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        # ffn/mlp
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=hidden_dim,
            out_features=embed_dim,
            drop=drop
        )

    def forward(self, x):
        # phase 1
        x = self.norm1(x)
        attn_out = self.attention(x)
        x = x + attn_out # residual connection
        # phase 2
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out # residual connection

        return x