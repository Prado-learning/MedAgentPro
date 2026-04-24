"""
Minimal Glaucoma segmentation wrappers around MSA.

Main output:
- 0/1 masks at the requested save path

Optional outputs (when GLAUCOMA_SAVE_MASK_VIS=1):
- *_vis.png      : 0/255 visualization mask
- *_overlay.png  : overlay on fundus image
- *_msa_raw.png  : direct file written by MSA before 0/1 normalization
"""

import os
import sys
import cv2
import numpy as np
from skimage import io

os.environ.setdefault(
    "GLAUCOMA_MASK_ROOT",
    "/root/MedAgentPro/data/REFUGE2/Annotation-Training400/Disc_Cup_Masks",
)

_msa_dir = "/root/MedAgentPro/tools/MSA"
if _msa_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(_msa_dir))

from tools.MSA.model import SAM_Adapter

_SAM_BASE = "/root/MedAgentPro/tools/MSA/checkpoint/sam/sam_vit_b_01ec64.pth"
_CUP_WEIGHTS = "/root/MedAgentPro/tools/MSA/Adapters/OpticCup_Fundus_SAM_1024.pth"
_DISC_WEIGHTS = "/root/MedAgentPro/tools/MSA/Adapters/OpticDisc_Fundus_SAM_1024.pth"

_disc_model = None
_cup_model = None


def _get_disc_model():
    global _disc_model
    if _disc_model is None:
        _disc_model = SAM_Adapter(sam_ckpt=_SAM_BASE, weights=_DISC_WEIGHTS)
    return _disc_model


def _get_cup_model():
    global _cup_model
    if _cup_model is None:
        _cup_model = SAM_Adapter(sam_ckpt=_SAM_BASE, weights=_CUP_WEIGHTS)
    return _cup_model


def _is_visual_save_enabled() -> bool:
    value = os.getenv("GLAUCOMA_SAVE_MASK_VIS", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_prompt_category(target: str) -> int:
    """
    Minimal explicit category control.

    Defaults assume REFUGE prompt mask labels:
    - cup  -> 0
    - disc -> 128
    """
    if target == "cup":
        return int(os.getenv("GLAUCOMA_PROMPT_CUP_VALUE", "0"))
    return int(os.getenv("GLAUCOMA_PROMPT_DISC_VALUE", "128"))


def _save_optional_visuals(image_path: str, save_path: str, mask01: np.ndarray, target: str):
    if not _is_visual_save_enabled():
        return

    vis_mask = (mask01 * 255).astype(np.uint8)
    vis_path = save_path.replace(".png", "_vis.png")
    io.imsave(vis_path, vis_mask, check_contrast=False)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return

    color = (0, 255, 0) if target == "disc" else (0, 0, 255)
    overlay = image.copy()
    overlay[mask01 > 0] = (
        0.65 * overlay[mask01 > 0] + 0.35 * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    overlay_path = save_path.replace(".png", "_overlay.png")
    cv2.imwrite(overlay_path, overlay)


def _predict_mask_minimal(model: SAM_Adapter, image_path: str, save_path: str, target: str):
    category = _get_prompt_category(target)
    raw_save_path = save_path
    if _is_visual_save_enabled():
        raw_save_path = save_path.replace(".png", "_msa_raw.png")
    return model.predict_mask(image_path, raw_save_path, category=category)


def segment_optic_disc(image_path: str, save_dir: str, save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    model = _get_disc_model()
    pred = _predict_mask_minimal(model, image_path, save_path, target="disc")

    mask = (pred > 0.5).astype(np.uint8)
    io.imsave(save_path, mask, check_contrast=False)
    _save_optional_visuals(image_path, save_path, mask, target="disc")
    return save_path


def segment_optic_cup(image_path: str, save_dir: str, save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    model = _get_cup_model()
    pred = _predict_mask_minimal(model, image_path, save_path, target="cup")

    mask = (pred > 0.5).astype(np.uint8)
    io.imsave(save_path, mask, check_contrast=False)
    _save_optional_visuals(image_path, save_path, mask, target="cup")
    return save_path