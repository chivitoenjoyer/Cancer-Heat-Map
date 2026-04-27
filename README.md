# Cancer-Heat-Map

Clasificacion de imagenes de cancer de mama usando **Vision Transformer (ViT)** con **PyTorch** y **Hugging Face Transformers/Datasets**.
El pipeline carga el dataset desde Hugging Face Hub, entrena un modelo ViT para **3 clases** y evalua el rendimiento generando un **reporte de clasificacion** y una **matriz de confusion**.

## Caracteristicas
- Dataset desde Hugging Face: `ShivamRaisharma/breastcancer`
- Modelo base: `google/vit-base-patch16-384`
- Clasificacion multiclase (3 etiquetas):
  - `0`: Benigno
  - `1`: Maligno
  - `2`: Normal
- Augmentations solo en entrenamiento y normalizacion con `torchvision.transforms`.
- Fine-tuning parcial: se entrena el **classifier** y las **ultimas 4 capas** del encoder de ViT.
- Evaluacion con:
  - `classification_report`
  - `confusion_matrix`, guardada como imagen.

## Estructura del repositorio
- `config.py`: configuracion central de dataset, modelo, hiperparametros, rutas y labels.
- `dataset.py`: carga del dataset de Hugging Face, transforms y dataloaders.
- `model_utils.py`: construccion y carga del modelo ViT.
- `train.py`: entrenamiento y guardado del mejor checkpoint.
- `evaluate.py`: evaluacion del checkpoint y generacion de metricas/figuras.
- `results/`: salidas generadas, por ejemplo `confusion_matrix.png`.

## Requisitos
Instala las dependencias con:

```bash
pip install -r requirements.txt
```

> Nota: el proyecto selecciona `cuda` si esta disponible.

## Configuracion
Ajusta parametros en `config.py`:
- `DATASET_NAME`: dataset de Hugging Face Hub.
- `MODEL_NAME`: checkpoint del ViT.
- `IMAGE_SIZE`: 384.
- `NUM_LABELS`, `ID2LABEL`, `LABEL2ID`.
- Hiperparametros: `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `WEIGHT_DECAY`.
- `CHECKPOINT_DIR`: carpeta de checkpoints, por defecto `./checkpoints`.
- `RESULTS_DIR`: carpeta de resultados, por defecto `./results`.

## Entrenamiento
El entrenamiento guarda el mejor modelo por accuracy de validacion en:

```bash
./checkpoints/best_model.pth
```

Comando recomendado:

```bash
python train.py --save_dir ./checkpoints --epochs 100 --batch_size 8 --lr 1e-6 --weight_decay 0.05
```

Argumentos disponibles:
- `--save_dir` (default: `./checkpoints`)
- `--epochs` (default: valor de `config.EPOCHS`)
- `--batch_size` (default: valor de `config.BATCH_SIZE`)
- `--lr` (default: valor de `config.LEARNING_RATE`)
- `--weight_decay` (default: valor de `config.WEIGHT_DECAY`)
- `--early_stopping_patience` (default: `10`): numero de epocas consecutivas sin mejora en `val_acc` tras las cuales se detiene el entrenamiento automaticamente. El mejor checkpoint ya estara guardado en ese momento.

## Evaluacion
`evaluate.py` carga el checkpoint:

```bash
./checkpoints/best_model.pth
```

y genera:
- reporte de clasificacion en consola
- matriz de confusion en `./results/confusion_matrix.png`

Ejecuta:

```bash
python evaluate.py
```

## Notas sobre el modelo
- Se usa `ViTForImageClassification.from_pretrained(..., ignore_mismatched_sizes=True, output_attentions=True)`.
- Se congelan todos los parametros y luego se habilita entrenamiento para:
  - `model.classifier`
  - `model.vit.encoder.layer[-4:]`
