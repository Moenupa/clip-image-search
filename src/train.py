import os

import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import CLIPModel
from .utils import (
    Transform,
    ImageTextDataset,
    AvgMeter,
    collate_fn,
    timestamp,
    CLIP_CHECKPOINT,
    DATA_ROOT,
    LOG_ROOT,
    MODEL_ROOT,
    DEVICE
)


def train_epoch(model, loader: DataLoader, optimizer) -> AvgMeter:
    loss_meter = AvgMeter()
    pbar = tqdm(loader, total=len(loader))
    for batch in pbar:
        optimizer.zero_grad()

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model(**batch, return_loss=True)

        loss = output.loss
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), batch['pixel_values'].size(0))

        pbar.set_postfix(train_loss=loss_meter.avg)
    return loss_meter


def eval_epoch(model, loader: DataLoader) -> AvgMeter:
    loss_meter = AvgMeter()
    pbar = tqdm(loader, total=len(loader))
    for batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        output = model(**batch, return_loss=True)
        loss = output.loss
        loss_meter.update(loss.item(), batch['pixel_values'].size(0))
        pbar.set_postfix(eval_loss=loss_meter.avg)
    return loss_meter


def train(checkpoint: str = CLIP_CHECKPOINT, lr: float = 2e-6, num_epochs: int = 5, batch_size: int = 32):
    dirname = f'{LOG_ROOT}/{timestamp()}'
    os.makedirs(dirname)
    
    train_loader = DataLoader(
        ImageTextDataset(DATA_ROOT, "train", transform=Transform(224, True)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    eval_loader = DataLoader(
        ImageTextDataset(DATA_ROOT, "eval", transform=Transform(224, False)),
        batch_size=16,
        collate_fn=collate_fn
    )
    
    model = CLIPModel.from_pretrained(checkpoint).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    logger = pd.DataFrame(columns=['train_loss', 'eval_loss'])
    logger.index.name = 'epoch'

    model.eval()
    with torch.no_grad():
        train_loss = eval_epoch(model, train_loader)
        eval_loss = eval_epoch(model, eval_loader)
        logger.loc[-1] = [train_loss.avg, eval_loss.avg]
        logger.to_csv(f'{dirname}/lr{lr}_b{batch_size}x{num_epochs}.csv', index=True)

    # os.makedirs(MODEL_ROOT, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer)

        model.save_pretrained(f'{dirname}/lr{lr}b{batch_size}x{epoch}/')
        model.eval()
        with torch.no_grad():
            eval_loss = eval_epoch(model, eval_loader)
        
        logger.loc[epoch] = [train_loss.avg, eval_loss.avg]
        logger.to_csv(f'{dirname}/lr{lr}_b{batch_size}x{num_epochs}.csv', index=True)


if __name__ == '__main__':
    for lr in [1e-6, 2e-6, 5e-6]:
        for batch_size in [32, 64]:
            train(lr=lr, batch_size=batch_size)
