"""
Multi Head Attention layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.CONFIG import EMBED_DIM, ATTENTION_HEADS

class Attention(nn.Module):
    def __init__(self, embed_dim = EMBED_DIM, num_heads = ATTENTION_HEADS, qkv_bias = False, attn_drop = 0, proj_drop = 0):
        super().__init__()
        self.num_heads = num_heads

        # ensure
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # dimensi per head
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5 # denominator untuk attention score (1/sqrt(d_k))

        # gabung Wq, Wk, Wv menjadi satu matriks besar
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # output proj, setelah attention
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # dapatkan bs, num_patches, embed_dim
        B, N, C = x.shape

        # qkv projection
        qkv = self.qkv(x)  # (B, N, 3*embed_dim)
        # reshape dan split menjadi q, k, v
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, num_heads, head_dim)
        # transpose untuk memudahkan perhitungan attention
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        # pisahkan q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, N, head_dim)

        # hitung atttention score
        # softmax(q*k.T) / scale
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
        attn = attn.softmax(dim = -1) # softmax over last dim (N)
        attn = self.attn_drop(attn) 

        # hitung weighted values
        x = (attn @ v)  # (B, num_heads, N, head_dim)

        # kembalikan ke bentuk semula -> gabung semua head
        x = x.transpose(1, 2).flatten(2)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x