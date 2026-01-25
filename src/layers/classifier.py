"""
MLP Head atau classifier untuk klasifikasi
Paper asli ViT menggunakan 1 layer mlp sebagai classifier (single linear layer)
"""

import torch
import torch.nn as nn
from config.CONFIG import EMBED_DIM, NUM_CLASSES

class VITClassifier(nn.Module):
    """
    Args:
        - embed_dim : dimensi embedding input (output dari transformer encoder)
        - num_classes : kelas untuk klasifikasi
        - hidden features : None karena satu layer, tapi bisa diisi (original paper tidak ada)
        - drop : dropout
    """
    def __init__(self, embed_dim = EMBED_DIM, num_classes = NUM_CLASSES, hidden_features = None, drop = 0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # single linear layer
        if hidden_features is None:
            self.head = nn.Sequential(
                nn.Dropout(drop),
                nn.Linear(embed_dim, num_classes)
            )
        # jika hidden_features diisi
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_features),
                nn.Tanh(),
                nn.Dropout(drop),
                nn.Linear(hidden_features, num_classes)
            )

    def forward(self, x):
        return self.head(x)
    
    # init weights
    def init_weights(self):
        for m in self.head.modules():
            if isinstance(m , nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)