import os
from functools import lru_cache


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MSA_ROOT = os.path.join(PROJECT_ROOT, "tools", "MSA")
DISC_WEIGHTS = os.path.join(MSA_ROOT, "Adapters", "OpticDisc_Fundus_SAM_1024.pth")
CUP_WEIGHTS = os.path.join(MSA_ROOT, "Adapters", "OpticCup_Fundus_SAM_1024.pth")
DEFAULT_SAM_CKPT = os.path.join(MSA_ROOT, "checkpoint", "sam", "sam_vit_b_01ec64.pth")

__all__ = ["segment_optic_disc", "segment_optic_cup"]


def _load_adapter(weights_path: str):
    try:
        from tools.MSA.model import SAM_Adapter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Glaucoma segmentation requires the local MSA dependencies. "
            f"Missing package: {exc.name}. Install torch, torchvision, opencv-python and related packages first."
        ) from exc

    sam_ckpt = os.getenv("SAM_CKPT_PATH", DEFAULT_SAM_CKPT)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MSA adapter weights not found: {weights_path}")
    if not os.path.exists(sam_ckpt):
        raise FileNotFoundError(
            "SAM checkpoint not found. Set SAM_CKPT_PATH to your sam_vit_b_01ec64.pth file."
        )

    return SAM_Adapter(sam_ckpt=sam_ckpt, weights=weights_path)


@lru_cache(maxsize=1)
def _disc_adapter():
    return _load_adapter(DISC_WEIGHTS)


@lru_cache(maxsize=1)
def _cup_adapter():
    return _load_adapter(CUP_WEIGHTS)


def segment_optic_disc(image_path: str, save_dir: str, save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    category = int(os.getenv("GLAUCOMA_DISC_CATEGORY", "1"))
    save_path = os.path.join(save_dir, save_name)
    return _disc_adapter().predict_mask(image_path, save_path, category=category)


def segment_optic_cup(image_path: str, save_dir: str, save_name: str):
    os.makedirs(save_dir, exist_ok=True)
    category = int(os.getenv("GLAUCOMA_CUP_CATEGORY", "1"))
    save_path = os.path.join(save_dir, save_name)
    return _cup_adapter().predict_mask(image_path, save_path, category=category)
