import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from config.CONFIG import (IMAGE_SIZE, PATCH_SIZE, NUM_CHANNELS, EMBED_DIM,
                           ATTENTION_HEADS, MLP_RATIO, HIDDEN_DIM, NUM_CLASSES,
                           DEPTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE)

# import model
from src.vit import VisionTransformer

print(f"Using device: {DEVICE}")

def data_loaders(batch_size):
    # transformasi data
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download mnist dataset
    train_dataset = datasets.MNIST(root='.\data', train=True, transform=transforms, download=True)
    test_dataset = datasets.MNIST(root='.\data', train=False, transform=transforms, download=True)

    # data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader