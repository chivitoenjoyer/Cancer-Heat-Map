import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
import dataset
import model_utils

torch.backends.cudnn.enabled = False


def generate_heatmap(image_path, checkpoint_path=None):
    device = torch.device(config.DEVICE)
    model = model_utils.load_trained_model(checkpoint_path)

    img = Image.open(image_path).convert("RGB")
    transform = dataset.get_val_transforms()
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = probs.argmax().item()
    pred_label = config.ID2LABEL[pred_id]
    confidence = probs[pred_id].item()

    # Attention del layer configurado: [1, num_heads, seq_len, seq_len]
    attentions = outputs.attentions[config.ATTENTION_LAYER_INDEX]
    att = attentions[0].mean(dim=0)   # promedio sobre cabezas → [seq, seq]
    cls_att = att[0, 1:].cpu().numpy()  # fila CLS sin el token CLS mismo → [num_patches]

    num_patches_side = config.IMAGE_SIZE // config.PATCH_SIZE  # 24 para 384/16
    att_map = cls_att.reshape(num_patches_side, num_patches_side)
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    att_resized = np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
    ) / 255.0

    img_display = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    img_gray = np.array(img_display.convert("L"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_display)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].imshow(att_resized, cmap="jet", alpha=config.HEATMAP_ALPHA)
    axes[1].set_title(f"Attention map — {pred_label} ({confidence:.1%})")
    axes[1].axis("off")

    fig.suptitle(f"Predicción: {pred_label} | Confianza: {confidence:.1%}", fontsize=13)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(config.RESULTS_DIR, f"heatmap_{basename}.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Predicción: {pred_label} (confianza: {confidence:.1%})")
    for i, label in config.ID2LABEL.items():
        print(f"  {label}: {probs[i].item():.1%}")
    print(f"Heatmap guardado en '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al checkpoint (default: checkpoints/best_model.pth)")
    args = parser.parse_args()

    generate_heatmap(args.image, args.checkpoint)
