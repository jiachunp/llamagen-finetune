import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import io
import glob
import pandas as pd
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        if 'ten_crop/' in feature_dir:
            aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
            aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
            if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
                self.aug_feature_dir = aug_feature_dir
                self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        # self.feature_files = sorted(os.listdir(feature_dir))
        # self.label_files = sorted(os.listdir(label_dir))
        # TODO: make it configurable
        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        
        # aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
        # features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)

class ParquetImageNetDataset(Dataset):
    def __init__(self, parquet_dir, transform=None):
        self.parquet_files = sorted(glob.glob(f"{parquet_dir}/train-*.parquet"))
        self.data = []
        for file in self.parquet_files:
            df = pd.read_parquet(file)
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        image_bytes = row['image']['bytes']
        label = int(row['label'])
        # If image_bytes is a string, decode base64, else use bytes directly
        if isinstance(image_bytes, str):
            import base64
            image_bytes = base64.b64decode(image_bytes)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def build_imagenet(args, transform):
    return ParquetImageNetDataset(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
        f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)