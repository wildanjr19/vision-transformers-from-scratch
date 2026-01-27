"""
Gabungkan semua layers untuk menjadi ViT.
    - cls token -> belajar merangkum seluruh gambar
    - positional embedding -> informasi posisi patch
1. Ekstraksi fitur : dapat cls token
2. Klasifikasi : cls token masuk ke classifier
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.CONFIG import EMBED_DIM, IMAGE_SIZE, PATCH_SIZE, NUM_CLASSES, NUM_CHANNELS, DEPTH, ATTENTION_HEADS, HIDDEN_DIM
from src.layers.patch_embedding import PatchEmbedding
from src.layers.transformer_encoder import TransformerEncoderBlock
from src.layers.classifier import VITClassifier

class VisionTransformer(nn.Module):
    """
    Args:
        - img_size : ukuran gambar input (IMAGE_SIZE)
        - patch_size : ukuran patch (PATCH_SIZE)
        - in_channels : input channels (NUM_CHANNELS)
        - num_classes : kelas untuk klasifikasi (NUM_CLASSES)
        - embed_dim : dimensi embedding (EMBED_DIM)
        - depth : jumlah transformer encoder block (DEPTH)
        - num_heads : attention heads (ATTENTION_HEADS)
        - mlp_ratio : rasio di hidden dim mlp transformer encoder (HIDDEN_DIM)
        - qkv_bias : tambah bias?
        - drop_rate : dropout rate
        - attn_drop_rate : attention dropout rate
    """
    def __init__(self,
                 img_size = IMAGE_SIZE,
                 patch_size = PATCH_SIZE,
                 in_channels = NUM_CHANNELS,
                 num_classes = NUM_CLASSES,
                 embed_dim = EMBED_DIM,
                 depth = DEPTH,
                 num_heads = ATTENTION_HEADS,
                 mlp_ratio = HIDDEN_DIM,
                 qkv_bias = True,
                 drop_rate = 0,
                 attn_drop_rate = 0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # get num pathces
        num_patches = self.patch_embed.num_pathces

        # 2. cls & pos token (learnable param)
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # [1, 1, embed_dim]
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim)) # [1, num_patches + 1, embed_dim]
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. transformer encoder block, sebanyak depth kali
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])

        # 4. layer norm terakhir sebelum classifier
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # 5. classifier / mlp head
        self.head = VITClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            drop=drop_rate
        )

        # init weights
        self._init_weights()

    # helper init weights
    def _init_weights(self):
        # init cls token dan pos embed
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # semua layer -> loop di module
        for m in self.modules():
            if isinstance(m , nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)

    def forward_features(self, x):
        """
        Untuk ekstraksi fitur, sehingga mendapatkan embedding dari cls token
        """
        # bs
        bs = x.shape[0]

        # 1. ubah gambar jadi sequence patch
        x = self.patch_embed(x) # [bs, num_patches, embed_dim]

        # 2. tambahkan cls token di awal seq
        # duplikasi sebanyak bs
        cls_tokens = self.cls_token.expand(bs, -1, -1) # [bs, 1, embed_dim]
        # gabungkan dengan patch
        x = torch.cat((cls_tokens, x), dim=1) # [bs, 1 + num_patches, embed_dim]

        # 3. tambahkan pos embedding
        x = x + self.pos_embed # [bs, 1 + num_patches, embed_dim] tidak berubah (broadcasting otomatis)
        x = self.pos_drop(x)

        # 4. masuk ke transformer encoder block
        x = self.blocks(x) # [bs, 1 + num_patches, embed_dim]

        # 5. final norm
        x = self.norm(x) # [bs, 1 + num_patches, embed_dim]

        # 6. ambil hanya cls token saja (token index 0)
        return x[:, 0] # [bs, embed_dim]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x