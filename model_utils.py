import os

import torch
from transformers import ViTForImageClassification

import config


def get_vit_model():
    model = ViTForImageClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
        output_attentions=True,
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True

    return model.to(config.DEVICE)


def get_best_model_path(save_dir=config.CHECKPOINT_DIR):
    return os.path.join(save_dir, config.BEST_MODEL_FILENAME)


def load_trained_model(path=None):
    checkpoint_path = path or get_best_model_path()
    model = get_vit_model()
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()
    return model
