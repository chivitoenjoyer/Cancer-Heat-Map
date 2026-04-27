import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import config


class BreastCancerDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

        all_cols = hf_dataset.column_names
        self.label_col = 'label' if 'label' in all_cols else [c for c in all_cols if c != 'image'][0]
        self.label2id = config.LABEL2ID  # fixed mapping from config, not derived from the dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item['image'].convert("RGB")
        pixel_values = self.transform(img)
        label = torch.tensor(self.label2id[item[self.label_col]], dtype=torch.long)
        return {
            'pixel_values': pixel_values,
            'label': label
        }


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def create_dataloaders(batch_size=None, num_workers=0):
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    print(f"Cargando dataset desde el Hub: {config.DATASET_NAME}...")
    raw_dataset = load_dataset(config.DATASET_NAME)

    if 'test' in raw_dataset:
        train_raw = raw_dataset['train']
        val_raw = raw_dataset['test']
    else:
        split = raw_dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_raw = split['train']
        val_raw = split['test']

    train_dataset = BreastCancerDataset(train_raw, get_train_transforms())
    val_dataset = BreastCancerDataset(val_raw, get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset listo.")
    return train_loader, val_loader
