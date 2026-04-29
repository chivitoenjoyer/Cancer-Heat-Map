import os
import uuid

import numpy as np
import torch
from PIL import Image
from matplotlib import cm

import config
import dataset
import model_utils

torch.backends.cudnn.enabled = False


def _load_checkpoint_meta(checkpoint_path: str) -> dict:
    try:
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        return {
            "epoch": ckpt.get("epoch"),
            "val_acc": ckpt.get("val_acc"),
            "val_loss": ckpt.get("val_loss"),
        }
    except Exception:
        return {}


_MODEL = None
_TRANSFORM = None
_CKPT_META = None


def get_model_bundle(checkpoint_path: str | None = None):
    global _MODEL, _TRANSFORM, _CKPT_META

    ckpt_path = checkpoint_path or model_utils.get_best_model_path()
    if _MODEL is None:
        _MODEL = model_utils.load_trained_model(ckpt_path)
        _TRANSFORM = dataset.get_val_transforms()
        _CKPT_META = _load_checkpoint_meta(ckpt_path)

    return _MODEL, _TRANSFORM, _CKPT_META, ckpt_path


def predict_with_heatmap(pil_image: Image.Image, checkpoint_path: str | None = None) -> dict:
    device = torch.device(config.DEVICE)
    model, transform, meta, ckpt_path = get_model_bundle(checkpoint_path)

    img_rgb = pil_image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(probs.argmax().item())

    attentions = outputs.attentions[config.ATTENTION_LAYER_INDEX]
    att = attentions[0].mean(dim=0)  # [tokens, tokens]
    cls_att = att[0, 1:].detach().cpu().numpy()

    num_patches_side = config.IMAGE_SIZE // config.PATCH_SIZE
    att_map = cls_att.reshape(num_patches_side, num_patches_side)
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

    att_resized = np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0

    img_resized = img_rgb.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    base_gray = img_resized.convert("L")
    base_rgb = Image.merge("RGB", (base_gray, base_gray, base_gray))

    heat_rgb = (cm.get_cmap("jet")(att_resized)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb)
    overlay = Image.blend(base_rgb, heat_img, alpha=float(config.HEATMAP_ALPHA))

    return {
        "checkpoint_path": ckpt_path,
        "pred_id": pred_id,
        "pred_label": config.ID2LABEL[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probs": {config.ID2LABEL[i]: float(probs[i].item()) for i in range(config.NUM_LABELS)},
        "model_val_acc": meta.get("val_acc"),
        "model_epoch": meta.get("epoch"),
        "model_val_loss": meta.get("val_loss"),
        "original": img_resized,
        "overlay": overlay,
    }


def save_result_images(result: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    job_id = uuid.uuid4().hex

    orig_name = f"{job_id}_original.png"
    overlay_name = f"{job_id}_heatmap.png"

    result["original"].save(os.path.join(output_dir, orig_name))
    result["overlay"].save(os.path.join(output_dir, overlay_name))

    return {"job_id": job_id, "orig_name": orig_name, "overlay_name": overlay_name}
