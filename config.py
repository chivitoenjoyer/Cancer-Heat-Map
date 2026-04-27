import torch

DATASET_NAME = "ShivamRaisharma/breastcancer"

CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_FILENAME = "best_model.pth"
RESULTS_DIR = "./results"

# Parametros del modelo
MODEL_NAME = "google/vit-base-patch16-384"
IMAGE_SIZE = 384
PATCH_SIZE = 16
NUM_LABELS = 3
ID2LABEL = {0: "Benigno", 1: "Maligno", 2: "Normal"}
LABEL2ID = {"Benigno": 0, "Maligno": 1, "Normal": 2}

# Hiperparametros de entrenamiento
BATCH_SIZE = 8
LEARNING_RATE = 5e-6
EPOCHS = 100
WEIGHT_DECAY = 0.01

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

ATTENTION_LAYER_INDEX = -1
HEATMAP_ALPHA = 0.6
