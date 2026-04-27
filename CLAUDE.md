# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos principales

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo
python train.py

# Entrenar con early stopping personalizado
python train.py --early_stopping_patience 15

# Reanudar entrenamiento desde un checkpoint
python train.py --resume ./checkpoints/best_model.pth

# Evaluar el checkpoint guardado
python evaluate.py

# Generar heatmap de atención para una imagen
python heatmap.py --image ruta/imagen.jpg
python heatmap.py --image ruta/imagen.jpg --checkpoint ./checkpoints/best_model.pth
```

## Arquitectura

Pipeline de clasificación de imágenes de cáncer de mama con ViT (`google/vit-base-patch16-384`) en 3 clases: `Benigno (0)`, `Maligno (1)`, `Normal (2)`.

**Flujo de datos:**
`config.py` → `dataset.py` → `train.py` → `checkpoints/best_model.pth` → `evaluate.py` / `heatmap.py` → `results/`

**Módulos:**
- `config.py` — fuente única de verdad: rutas, hiperparámetros, labels y device. Modificar aquí antes de tocar otros archivos.
- `dataset.py` — carga `ShivamRaisharma/breastcancer` desde Hugging Face Hub. Las transforms de entrenamiento aplican Grayscale antes de ColorJitter (orden importante). Las de validación solo resize + grayscale + normalize.
- `model_utils.py` — construye el ViT con fine-tuning parcial: solo `classifier` y las últimas 4 capas del encoder (`vit.encoder.layer[-4:]`) tienen `requires_grad=True`. `load_trained_model()` carga desde checkpoint y pone el modelo en `eval()`. Lanza `FileNotFoundError` con mensaje claro si el checkpoint no existe.
- `train.py` — loop de entrenamiento con `ReduceLROnPlateau` (patience=5, min_lr=1e-8) y early stopping (`--early_stopping_patience`, default 10). Guarda checkpoint completo (incluye `scheduler_state_dict` y `args`) solo cuando mejora `val_acc`. Escribe métricas por época en `checkpoints/training_log.csv`. Soporta `--resume` para continuar desde un checkpoint.
- `evaluate.py` — carga `checkpoints/best_model.pth` y genera reporte de clasificación (guardado en `results/classification_report.txt`) + `results/confusion_matrix.png` + análisis de seguridad para las 3 clases. Puede ejecutarse de forma independiente sin re-entrenar.
- `heatmap.py` — genera visualización de attention maps del ViT sobre una imagen de entrada. Usa `output_attentions=True` del modelo, promedia sobre las cabezas de atención del último layer (`ATTENTION_LAYER_INDEX`) y superpone el mapa sobre la imagen con `HEATMAP_ALPHA`. Guarda en `results/heatmap_<nombre>.png` e imprime predicción con confianza por clase.

## Notas importantes

- El dataset se descarga automáticamente de Hugging Face Hub en la primera ejecución.
- El modelo base (`google/vit-base-patch16-384`) también se descarga de Hugging Face en la primera ejecución (~350 MB).
- El checkpoint entrenado (`best_model.pth`) pesa ~545 MB y se gestiona con Git LFS (`*.pth` en `.gitattributes`).
- `torch.backends.cudnn.enabled = False` se establece en `main()` de `train.py` y a nivel de módulo en `evaluate.py` y `heatmap.py` para garantizar reproducibilidad; no eliminarlo sin revisar el impacto.
