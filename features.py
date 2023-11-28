import os
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPProcessor, CLIPModel
from utils.data import Transform, ImageTextDataset, collate_fn


# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("./out/lr2e-06_3").to(device)
process = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

test_loader = DataLoader(
    ImageTextDataset('data', "valid", transform=Transform(224, False)),
    batch_size=1,
    collate_fn=lambda x: collate_fn(x, process.tokenizer),
)


# Function that computes the feature vectors for a batch of images
def image_features(batch) -> np.ndarray:
    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        features = model.get_image_features(batch['pixel_values'])
        features = F.normalize(features, dim=-1)

    return features.cpu().numpy()


def text_features(captions) -> np.ndarray:
    with torch.no_grad():
        inputs = process.tokenizer(captions, max_length=32, padding="max_length", return_tensors="pt", truncation=True)
        features = model.get_text_features(**inputs)

    return features.cpu().numpy()


def save_image_features(save_name: str = "25k_features.npy") -> None:
    all_features = []
    for batch in tqdm(test_loader):
        features = text_features(batch)
        all_features.append(features)
    all_features = np.stack(all_features)
    np.save(save_name, all_features)


if __name__ == "__main__":
    # sim = (text_features @ image_features.T).squeeze(0)
    # order = sim.argsort(descending=True)
    save_image_features()
