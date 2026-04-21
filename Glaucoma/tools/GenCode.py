# Generated code


def compute_cup_to_disc_ratio_3(inputs, save_dir, save_name):
    """
    Compute the cup-to-disc ratio from binary segmentation masks.
    Inputs is a list: [disc_mask, cup_mask], each either a numpy array or image file path.
    Writes the numerical ratio into a JSON file at save_dir/save_name under key 'step_3'.
    """
    import os
    import json
    import numpy as np
    from PIL import Image

    def load_mask(x):
        if isinstance(x, np.ndarray):
            mask = x
        else:
            mask = np.array(Image.open(x).convert('L'))
        if mask.ndim > 2:
            mask = mask[..., 0]
        return mask > 0

    disc_mask = load_mask(inputs[0])
    cup_mask = load_mask(inputs[1])

    disc_area = float(np.count_nonzero(disc_mask))
    cup_area = float(np.count_nonzero(cup_mask))
    ratio = cup_area / disc_area if disc_area > 0 else None

    path = os.path.join(save_dir, save_name)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            data = {}

    data['step_3'] = {'cup_to_disc_ratio': ratio}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return data['step_3']
