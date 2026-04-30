import argparse

import torch
from datasets import Features, Image as HFImage, Value, load_dataset
from datasets.exceptions import DatasetGenerationError
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
import dataset as dataset_utils
import model_utils

DEFAULT_EXTERNAL_DATASET = "gymprathap/Breast-Cancer-Ultrasound-Images-Dataset"


def normalize_label_name(label_name):
    return str(label_name).strip().lower().replace("_", " ").replace("-", " ")


def map_to_config_label_id(raw_label_name):
    normalized = normalize_label_name(raw_label_name)

    if normalized in {"benigno", "benign"}:
        return config.LABEL2ID["Benigno"]
    if normalized in {"maligno", "malignant"}:
        return config.LABEL2ID["Maligno"]
    if normalized in {"normal", "healthy"}:
        return config.LABEL2ID["Normal"]

    if "benig" in normalized:
        return config.LABEL2ID["Benigno"]
    if "malig" in normalized:
        return config.LABEL2ID["Maligno"]
    if "normal" in normalized or "health" in normalized:
        return config.LABEL2ID["Normal"]

    raise ValueError(f"No se pudo mapear la clase '{raw_label_name}' a config.LABEL2ID.")


def resolve_label_col(split):
    cols = split.column_names
    for candidate in ("label", "labels", "class", "target", "diagnosis", "category"):
        if candidate in cols:
            return candidate

    fallback = [c for c in cols if c.lower() not in {"image", "img", "pixel_values"}]
    if not fallback:
        raise ValueError(f"No se encontro columna de etiqueta en {cols}.")
    return fallback[0]


def extract_image_path(raw_image):
    if isinstance(raw_image, dict):
        return raw_image.get("path")
    return getattr(raw_image, "filename", None) or None


def load_external_dataset(dataset_name):
    try:
        return load_dataset(dataset_name)
    except DatasetGenerationError as err:
        detailed_msg = f"{err} {err.__cause__ or ''}"
        if "Invalid string class label" not in detailed_msg:
            raise

        print(
            "Aviso: el dataset externo tiene etiquetas ClassLabel invalidas en el Hub. "
            "Se aplicara carga tolerante para mapear etiquetas desde texto/ruta."
        )
        fallback_features = Features({
            "image": HFImage(decode=False),
            "label": Value("string"),
        })
        return load_dataset(dataset_name, features=fallback_features)


def remove_mask_examples(split):
    if "image" not in split.column_names:
        return split, 0

    before = len(split)
    filtered = split.filter(
        lambda sample: "_mask" not in (extract_image_path(sample["image"]) or "").lower()
    )
    removed = before - len(filtered)
    return filtered, removed


class ExternalBreastDataset(Dataset):
    def __init__(self, hf_split, label_col, transform):
        self.hf_split = hf_split
        self.label_col = label_col
        self.transform = transform
        self.label_feature = hf_split.features.get(label_col)
        self.image_decoder = HFImage()

    def __len__(self):
        return len(self.hf_split)

    def _to_label_id(self, raw_label, raw_image):
        if isinstance(raw_label, torch.Tensor):
            raw_label = raw_label.item()

        if isinstance(raw_label, str):
            try:
                return map_to_config_label_id(raw_label)
            except ValueError as label_error:
                image_path = extract_image_path(raw_image)
                if image_path:
                    return map_to_config_label_id(image_path)
                raise label_error

        if isinstance(raw_label, int):
            if hasattr(self.label_feature, "names") and self.label_feature.names:
                if raw_label < 0 or raw_label >= len(self.label_feature.names):
                    raise ValueError(f"Indice de clase fuera de rango: {raw_label}")
                label_name = self.label_feature.names[raw_label]
                return map_to_config_label_id(label_name)
            if raw_label in config.ID2LABEL:
                return raw_label
            raise ValueError(
                "Etiqueta entera sin nombres de clase; no se puede garantizar mapeo correcto a config.py."
            )

        raise ValueError(f"Tipo de etiqueta no soportado: {type(raw_label).__name__}")

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        raw_image = item["image"]
        if isinstance(raw_image, dict):
            image = self.image_decoder.decode_example(raw_image).convert("RGB")
        elif isinstance(raw_image, Image.Image):
            image = raw_image.convert("RGB")
        else:
            raise ValueError(f"Formato de imagen no soportado: {type(raw_image).__name__}")

        x = self.transform(image)
        y = torch.tensor(self._to_label_id(item[self.label_col], raw_image), dtype=torch.long)
        return {"pixel_values": x, "label": y}


def get_validation_split(raw_dataset):
    if "validation" in raw_dataset:
        return raw_dataset["validation"], "validation"
    if "test" in raw_dataset:
        return raw_dataset["test"], "test"
    if "train" in raw_dataset:
        split = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
        return split["test"], "train_test_split(test=0.2)"
    raise ValueError("El dataset no contiene split usable (validation/test/train).")


def main(args):
    if args.dataset_name == config.DATASET_NAME:
        raise ValueError(
            "Debes usar un dataset diferente al de entrenamiento para esta evaluacion de generalizacion."
        )

    device = torch.device(config.DEVICE)
    raw_dataset = load_external_dataset(args.dataset_name)
    val_split, split_name = get_validation_split(raw_dataset)
    val_split, removed_masks = remove_mask_examples(val_split)
    label_col = resolve_label_col(val_split)

    val_dataset = ExternalBreastDataset(
        hf_split=val_split,
        label_col=label_col,
        transform=dataset_utils.get_val_transforms(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    model = model_utils.load_trained_model(args.checkpoint)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inferencia validacion externa", unit="batch"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            logits = model(inputs).logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("El split de validacion externo esta vacio.")

    accuracy = 100.0 * correct / total
    class_order = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]

    print(f"Dataset externo: {args.dataset_name}")
    print(f"Split usado: {split_name}")
    if removed_masks > 0:
        print(f"Mascaras removidas del split: {removed_masks}")
    print(f"Columna de etiqueta: {label_col}")
    print(f"Clases objetivo (config.py): {class_order}")
    print(f"Aciertos: {correct}/{total}")
    print(f"Tasa de aciertos (accuracy): {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Calcula la tasa de aciertos del mejor modelo sobre un split de validacion "
            "de un dataset externo de Hugging Face, mapeando sus clases a config.py."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_EXTERNAL_DATASET,
        help=(
            "Dataset externo de Hugging Face (debe ser distinto a config.DATASET_NAME). "
            f"Default: {DEFAULT_EXTERNAL_DATASET}"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint. Por defecto usa ./checkpoints/best_model.pth",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size para inferencia (default: config.BATCH_SIZE).",
    )
    args = parser.parse_args()
    main(args)
