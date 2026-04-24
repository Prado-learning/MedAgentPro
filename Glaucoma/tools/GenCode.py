# Generated code


def compute_cup_to_disc_ratio_using_segmentation_masks_3(inputs, save_dir, save_name):
    """
    Compute cup-to-disc ratio (vertical) from segmentation masks.
    inputs[0]: optic disc mask (file path or numpy array), values {0,1}
    inputs[1]: optic cup mask (file path or numpy array), values {0,1}
    Writes the numerical result into JSON at os.path.join(save_dir, save_name).
    """
    import os, json
    import numpy as np
    from PIL import Image

    def load_mask(x):
        if isinstance(x, str):
            arr = np.array(Image.open(x).convert("L"))
            return (arr > 0).astype(np.uint8)
        arr = np.array(x)
        return (arr > 0).astype(np.uint8)

    disc_mask = load_mask(inputs[0])
    cup_mask = load_mask(inputs[1])

    def vertical_extent(mask):
        ys = np.where(mask > 0)[0]
        if ys.size == 0:
            return 0.0
        return float(ys.max() - ys.min() + 1)

    disc_height = vertical_extent(disc_mask)
    cup_height = vertical_extent(cup_mask)

    cdr = float(cup_height / disc_height) if disc_height > 0 else 0.0

    out_path = os.path.join(save_dir, save_name)
    data = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = {}

    data["compute_cup_to_disc_ratio_using_segmentation_masks_3"] = {"cup_to_disc_ratio": cdr}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return data["compute_cup_to_disc_ratio_using_segmentation_masks_3"]
