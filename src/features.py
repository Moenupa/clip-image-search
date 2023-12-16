import os
import glob

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel
from utils import (
    Transform,
    ImageTextDataset,
    collate_fn,
    timestamp,
    DEVICE,
    MODEL_ROOT,
    LOG_ROOT
)


def get_sim_scores(model, loader: DataLoader) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            score = outputs.logits_per_text.diagonal().cpu().numpy()
            scores.append(score)
    return np.block(scores)


def save_image_features(model, loader: DataLoader, path: str = "features_25k.npy") -> None:
    all_features = []
    with torch.no_grad():
        for batch in tqdm(loader):
            feat_img = model.get_image_features(batch['pixel_values'].to(DEVICE))
            feat_img = F.normalize(feat_img, dim=-1)
            features = feat_img.cpu().numpy()
            all_features.append(features)
    all_features = np.stack(all_features).squeeze()
    np.save(path, all_features)


def compute_features(model_name: str, report_score: bool = True) -> None:
    if '/' not in model_name:
        # fetch model locally
        model_path = f'{MODEL_ROOT}/{model_name}'
        assert os.path.exists(model_path), f'model path {model_path} does not exist'
    else:
        model_path = model_name

    valid_loader = DataLoader(
        ImageTextDataset('data', "valid", transform=Transform(224, False)),
        batch_size=1,
        collate_fn=collate_fn
    )

    dirname = f'{LOG_ROOT}/{timestamp()}'
    os.makedirs(dirname)
    model = CLIPModel.from_pretrained(model_path).to(DEVICE)
    save_image_features(model, valid_loader, f'{dirname}/{model_name}.npy')
    
    if report_score:
        test_loader = DataLoader(
            ImageTextDataset('data', "test", transform=Transform(224, False)),
            batch_size=1,
            collate_fn=collate_fn
        )
        np.savetxt(f'{dirname}/scores.csv', get_sim_scores(model, test_loader))


if __name__ == "__main__":
    compute_features("lr2e-06_b16x8")