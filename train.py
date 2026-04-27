import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cudnn.enabled = False

import config
import dataset
import model_utils

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress = tqdm(dataloader, desc=f'Epoch {epoch}', unit='batch')

    for batch in progress:
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        
        # Métricas de clasificación
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{100.*correct/total:.1f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', unit='batch'):
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

def main(args):
    torch.cuda.manual_seed(33)
    np.random.seed(33)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(config.DEVICE)

    train_loader, val_loader = dataset.create_dataloaders(batch_size=args.batch_size)

    model = model_utils.get_vit_model()

    for param in model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True

    weights = torch.tensor([1.0, 2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0

    print(f"Iniciando entrenamiento")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc
                }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Guardado con Val Acc: {best_val_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
