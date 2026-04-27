import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
import dataset
import model_utils

torch.backends.cudnn.enabled = False


def evaluate_model():
    device = torch.device(config.DEVICE)
    _, val_loader = dataset.create_dataloaders()
    model = model_utils.load_trained_model(model_utils.get_best_model_path())

    all_preds = []
    all_labels = []

    print("Iniciando evaluacion...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs).logits
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]

    report = classification_report(all_labels, all_preds, target_names=target_names)
    print(report)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Reporte guardado en '{report_path}'")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Prediccion")
    plt.ylabel("Realidad")
    plt.title("Matriz de Confusion - Modelo ViT Mama")

    output_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Matriz de confusion guardada en '{output_path}'")

    # cm[real][pred] — análisis de seguridad para las 3 clases
    mal = config.LABEL2ID["Maligno"]
    ben = config.LABEL2ID["Benigno"]
    nor = config.LABEL2ID["Normal"]

    print("\n--- Analisis de Seguridad ---")
    print(f"Maligno clasificado como Benigno : {cm[mal][ben]}")
    print(f"Maligno clasificado como Normal  : {cm[mal][nor]}")
    print(f"Benigno clasificado como Maligno : {cm[ben][mal]}")
    print(f"Normal  clasificado como Maligno : {cm[nor][mal]}")


if __name__ == "__main__":
    evaluate_model()
