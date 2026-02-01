"""
Konfigurasi untuk ViT
"""
import torch

IMAGE_SIZE = 28
PATCH_SIZE = 4
NUM_CHANNELS = 1
EMBED_DIM = 64
ATTENTION_HEADS = 4
MLP_RATIO = 4
HIDDEN_DIM = EMBED_DIM * MLP_RATIO
NUM_CLASSES = 10
DEPTH = 3
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"