import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    CenterCrop,
    ToDtype,
    Normalize,
    Resize
)

import pandas as pd
from typing import Optional, Callable


def collate_fn(examples):
    # text to tokenizer, images to numpy
    pixel_values = torch.stack([example[0] for example in examples])
    captions = [example[1] for example in examples]
    inputs = tokenizer(captions, max_length=32, padding="max_length", return_tensors="pt", truncation=True)
    return {
        "pixel_values": pixel_values,
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask']
    }


class Transform(torch.nn.Module):
    def __init__(self, image_size, augment_images):
        super().__init__()
        if augment_images:
            crop_size = int(image_size * 0.8)
            self.transforms = Compose([
                RandomResizedCrop(size=(224, 224), antialias=True),
                RandomHorizontalFlip(p=0.5),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.transforms = torch.nn.Sequential(
                Resize(image_size),
                CenterCrop(image_size),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
        return x


class ImageTextDataset(VisionDataset):
    """
    Dataset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored.
            The expected format is jsonlines where each line is a json object containing to keys.
            `filename`: The path to the image.
            `captions`: An `array` of captions.
        split: (string): Dataset split name. Is used for parsing tsv files from `root` folder.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            split: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        path = f"meta/{split}.tsv"

        self.captions = []
        self.image_paths = []
        for i, example in pd.read_csv(path, sep='\t').iterrows():
            self.captions.append(example["ai_description"])
            self.image_paths.append(example["photo_id"])

    def _load_image(self, idx: int):
        path = f"{self.root}/{self.image_paths[idx]}.jpg"
        return read_image(path, mode=ImageReadMode.RGB)

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.captions)
