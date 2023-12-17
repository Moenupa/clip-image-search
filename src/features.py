import os
import glob

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel
from .utils import (
    Transform,
    ImageTextDataset,
    load_model,
    collate_fn,
    timestamp,
    DEVICE,
    MODEL_ROOT,
    LOG_ROOT,
    CLIP_CHECKPOINT
)


def get_sim_scores(model: CLIPModel, loader: DataLoader) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            score = outputs.logits_per_text.diagonal().cpu().numpy()
            scores.append(score)
    return np.block(scores)


def save_image_features(model: CLIPModel, loader: DataLoader, path: str = "features_25k.npy") -> None:
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
    valid_loader = DataLoader(
        ImageTextDataset('data', "valid", transform=Transform(224, False)),
        batch_size=1,
        collate_fn=collate_fn
    )

    dirname = f'{LOG_ROOT}/{timestamp()}'
    savename = model_name.replace("/", "-").replace("\\", "-")
    os.makedirs(dirname)
    model = load_model(model_name).to(DEVICE)
    save_image_features(model, valid_loader, f'{dirname}/{savename}.npy')
    
    if report_score:
        test_loader = DataLoader(
            ImageTextDataset('data', "test", transform=Transform(224, False)),
            batch_size=1,
            collate_fn=collate_fn
        )
        scores = get_sim_scores(model, test_loader)
        np.savetxt(f'{dirname}/{savename}_scores.csv', scores)
        print(f'{model_name} score: {np.average(scores)}')
        
        
def gen_out_symlinks():
    os.makedirs(f'{MODEL_ROOT}', exist_ok=True)
    for model_dir in glob.glob('log/*/*/'):
        model_name = os.path.basename(model_dir[:-1])
        if os.path.exists(f'{MODEL_ROOT}/{model_name}'):
            continue
        os.symlink(
            os.path.abspath(model_dir), 
            f'{MODEL_ROOT}/{model_name}', 
            target_is_directory=True
        )
        

def gen_symlinks(regex="log/*/*.npy"):
    os.makedirs(f'tmp', exist_ok=True)
    for file_path in glob.glob(regex):
        file_name = os.path.basename(file_path)
        if os.path.exists(f'tmp/{file_name}'):
            continue
        os.symlink(
            os.path.abspath(file_path), 
            f'tmp/{file_name}'
        )
        

def print_scores():
    for f in glob.glob('log/*/*_scores.csv'):
        scores = np.loadtxt(f)
        print(f, np.average(scores))


if __name__ == "__main__":
    print_scores()
    exit(0)
    
    gen_out_symlinks()
    model_names = [CLIP_CHECKPOINT]
    for model_name in model_names:
        compute_features(model_name)
    gen_symlinks()