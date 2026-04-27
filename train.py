import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
import dataset
import model_utils


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

    for batch in progress:
        inputs = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    return total_loss / len(dataloader), 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100.0 * correct / total


def main(args):
    torch.manual_seed(33)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(33)
    np.random.seed(33)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(config.DEVICE)
    train_loader, val_loader = dataset.create_dataloaders(batch_size=args.batch_size)
    model = model_utils.get_vit_model()

    weights = torch.tensor([1.0, 2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-8,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint no encontrado: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_acc = ckpt["val_acc"]
        start_epoch = ckpt["epoch"]
        print(f"Reanudando desde época {start_epoch} con Val Acc: {best_val_acc:.2f}%")

    log_path = os.path.join(args.save_dir, "training_log.csv")
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if not log_exists:
        csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    print("Iniciando entrenamiento")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.2e}"
        )
        csv_writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.2f}",
            f"{val_loss:.4f}",
            f"{val_acc:.2f}",
            f"{current_lr:.2e}",
        ])
        log_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": best_val_acc,
                "args": vars(args),
            }, model_utils.get_best_model_path(args.save_dir))
            print(f"Guardado con Val Acc: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping en época {epoch} (sin mejora por {args.early_stopping_patience} épocas consecutivas)")
                break

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=config.CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Ruta al checkpoint para reanudar el entrenamiento")

    args = parser.parse_args()
    main(args)
