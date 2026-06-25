import torch
import faiss
import numpy as np

from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Subset

from models.feature_extractor import (
    load_feature_extractor,
    extract_features
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = load_feature_extractor().to(device)

transform = ResNet18_Weights.DEFAULT.transforms()

dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Use only the first 10,000 images
dataset = Subset(
    dataset,
    range(10000)
)

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False
)

all_features = []

print("Extracting features...")

with torch.no_grad():

    total_batches = len(loader)

    for batch_num, (images, labels) in enumerate(loader):

        print(
            f"Processing batch {batch_num+1}/{total_batches}"
        )

        features = extract_features(
            model,
            images,
            device
        )

        all_features.append(
            features.cpu()
        )

features = torch.cat(
    all_features,
    dim=0
)

features = (
    features.numpy()
    .astype(np.float32)
)

print("Building FAISS index...")

index = faiss.IndexFlatL2(512)

index.add(features)

faiss.write_index(
    index,
    "outputs/faiss.index"
)

print("FAISS index saved.")
print("Vectors stored:", index.ntotal)