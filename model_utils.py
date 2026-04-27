import torch
from transformers import ViTForImageClassification, ViTConfig
import config

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
        output_attentions=True
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    for param in model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True

    return model.to(config.DEVICE)


def save_model(model, path=config.CHECKPOINT_DIR):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    
    model.save_pretrained(path)
    print(f"Modelo guardado en: {path}")

def load_trained_model(path=config.CHECKPOINT_DIR):
    model = ViTForImageClassification.from_pretrained(
        path,
        output_attentions=True
    )
    model.to(config.DEVICE)
    return model

