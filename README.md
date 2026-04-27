# Cancer-Heat-Map

Clasificación de imágenes de cáncer de mama usando **Vision Transformer (ViT)** con **PyTorch** + **Hugging Face Transformers/Datasets**.  
El pipeline carga el dataset desde Hugging Face Hub, entrena un modelo ViT para **3 clases** y evalúa el rendimiento generando un **reporte de clasificación** y una **matriz de confusión**.

## Características
- Dataset desde Hugging Face: `ShivamRaisharma/breastcancer`
- Modelo base: `google/vit-base-patch16-384`
- Clasificación multiclase (3 etiquetas):
  - `0`: Benigno
  - `1`: Maligno
  - `2`: Normal
- Augmentations y normalización con `torchvision.transforms`
- Fine-tuning parcial: se entrena el **classifier** y las **últimas 4 capas** del encoder de ViT
- Evaluación con:
  - `classification_report`
  - `confusion_matrix` (se guarda como imagen)

## Estructura del repositorio
- `config.py`: configuración central (dataset, modelo, hiperparámetros, device, labels).
- `dataset.py`: carga del dataset (HF `datasets`) + transforms + dataloaders.
- `model_utils.py`: construcción/carga/guardado del modelo ViT.
- `train.py`: entrenamiento y guardado del mejor checkpoint.
- `evaluate.py`: evaluación del checkpoint y generación de métricas/figuras.
- `results/`: salidas generadas (ej. `confusion_matrix.png`).

## Requisitos
Dependencias principales (instálalas con pip):
- `torch`, `torchvision`
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `scikit-learn`
- `matplotlib`, `seaborn`

> Nota: el proyecto selecciona `cuda` si está disponible (ver `config.py`).

## Configuración
Ajusta parámetros en `config.py`:
- `DATASET_NAME`: dataset de Hugging Face Hub.
- `MODEL_NAME`: checkpoint del ViT.
- `IMAGE_SIZE`: 384
- `NUM_LABELS`, `ID2LABEL`, `LABEL2ID`
- Hiperparámetros: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `WEIGHT_DECAY`
- `CHECKPOINT_DIR`: carpeta de checkpoints (por defecto `./checkpoints/vit-breast-cancer`)

## Entrenamiento
El entrenamiento guarda el mejor modelo por **accuracy de validación** en un archivo `best_model.pth`.

Ejemplo:
```bash
python train.py --save_dir ./checkpoints --epochs 100 --lr 1e-6 --weight_decay 0.05
```

Argumentos disponibles (ver `train.py`):
- `--save_dir` (default: `./checkpoints`)
- `--epochs` (default: `100`)
- `--batch_size` (default: `8`)
- `--lr` (default: `1e-6`)
- `--weight_decay` (default: `0.05`)


## Evaluación
`evaluate.py` carga el checkpoint:
- `./checkpoints/best_model.pth`

y genera:
- reporte de clasificación en consola
- matriz de confusión en `confusion_matrix.png`

Ejecuta:
```bash
python evaluate.py
```

### Resultados
En este repo hay un ejemplo de salida:
- `results/confusion_matrix.png`

## Notas sobre el modelo
- Se usa `ViTForImageClassification.from_pretrained(..., ignore_mismatched_sizes=True, output_attentions=True)`.
- Se congelan todos los parámetros y luego se habilita entrenamiento para:
  - `model.classifier`
  - `model.vit.encoder.layer[-4:]`

