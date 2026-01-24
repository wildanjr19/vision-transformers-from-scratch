""""
Part Patch Embedding layer di ViT
Gambar input akan dipecah menjadi bagian/patch-patch kecil dan diubah menjadi representasi vektor (embedding).
Patch gambar ini tidak overlap, maka stride akan berperan disini.
Karena inputnya adalah gambar makan menggunakan Conv2D sebagai komponen utama disini.
"""
import torch
import torch.nn as nn

from config.CONFIG import IMAGE_SIZE, PATCH_SIZE, NUM_CHANNELS, EMBED_DIM


class PatchEmbedding:
    """
    Args:
        - in_channels: channels gambar input
        - embed_dim: dimensi embedding (output channels)
        - patch_size: ukuran patch (tinggi dan lebar) -> dihitung relative sesuai input
        - img_size: ukuran gambar input
    """
    def __init__(self, in_channels: int = NUM_CHANNELS, embed_dim: int = EMBED_DIM, patch_size: int = PATCH_SIZE, img_size: int = IMAGE_SIZE):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # dapatkan jumlah patch
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_pathces = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        x = self.proj(x) # [bs, embed_dim, grid_h, grid_w]
        # ratakan di H dan W, atau mulai dimensi kedua
        x = x.flatten(2) # [bs, embed_dim, num_patches]
        # req shape : [b, seq_len, embed_dim], jadi kita tukar dimensi 1 dan 2
        x = x.transpose(1, 2)
        return x