import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


class BreastCancerDataset(Dataset):
    def __init__(self, hf_dataset, transform, label_col=None, label2id=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

        all_cols = hf_dataset.column_names
        self.label_col = label_col or ("label" if "label" in all_cols else [c for c in all_cols if c != "image"][0])
        self.label2id = label2id or config.LABEL2ID

    def __len__(self):
        return len(self.hf_dataset)

    def _label_to_id(self, raw_label):
        if isinstance(raw_label, torch.Tensor):
            raw_label = raw_label.item()

        if isinstance(raw_label, int):
            return raw_label

        if isinstance(raw_label, str):
            if raw_label in self.label2id:
                return self.label2id[raw_label]
            if raw_label.isdigit():
                return int(raw_label)

        raise ValueError(f"Etiqueta no reconocida en columna '{self.label_col}': {raw_label!r}")

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item["image"].convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        y = torch.tensor(self._label_to_id(item[self.label_col]), dtype=torch.long)

        return {"pixel_values": x, "label": y}


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_dataloaders(batch_size=None):
    print(f"Cargando dataset desde el Hub: {config.DATASET_NAME}...")
    raw_dataset = load_dataset(config.DATASET_NAME)
    batch_size = batch_size or config.BATCH_SIZE

    if "test" in raw_dataset:
        train_raw = raw_dataset["train"]
        val_raw = raw_dataset["test"]
    else:
        split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
        train_raw = split["train"]
        val_raw = split["test"]

    train_dataset = BreastCancerDataset(train_raw, get_train_transforms())
    val_dataset = BreastCancerDataset(val_raw, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Dataset listo.")
    return train_loader, val_loader
