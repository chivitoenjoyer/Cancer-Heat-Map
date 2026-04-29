import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
import dataset
import model_utils


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_preds = []

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
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(predicted.detach().cpu().tolist())

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "f1": f"{100.0 * f1_score(all_labels, all_preds, average='macro', zero_division=0):.1f}%",
        })

    train_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(dataloader), train_f1


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs).logits
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(predicted.detach().cpu().tolist())

    val_f1 = 100.0 * f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(dataloader), val_f1


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

    weights = torch.tensor([1.0, 3.0, 1.0]).to(device)
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
    best_val_f1 = float("-inf")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint no encontrado: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_f1 = ckpt.get("val_f1", ckpt.get("val_acc", float("-inf")))
        best_val_loss = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt["epoch"]
        print(
            f"Reanudando desde época {start_epoch} con Val F1: {best_val_f1:.2f}% "
            f"y Val Loss: {best_val_loss:.4f}"
        )

    log_path = os.path.join(args.save_dir, "training_log.csv")
    log_exists = os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if not log_exists:
        csv_writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1", "lr"])

    print("Iniciando entrenamiento")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} F1: {train_f1:.2f}% | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.2f}% | LR: {current_lr:.2e}"
        )
        csv_writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_f1:.2f}",
            f"{val_loss:.4f}",
            f"{val_f1:.2f}",
            f"{current_lr:.2e}",
        ])
        log_file.flush()

        is_better_f1 = val_f1 > best_val_f1
        is_tie_f1 = np.isclose(val_f1, best_val_f1, rtol=0.0, atol=1e-6)
        is_better_loss_on_tie = is_tie_f1 and val_loss < best_val_loss

        if is_better_f1 or is_better_loss_on_tie:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": best_val_loss,
                "args": vars(args),
            }, model_utils.get_best_model_path(args.save_dir))
            print(f"Guardado con Val F1: {best_val_f1:.2f}% y Val Loss: {best_val_loss:.4f}")
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
