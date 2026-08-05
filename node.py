import base64
import io

import numpy as np
import torch
from PIL import Image


def _input_keys(kwargs, prefix="image"):
    """
    Return ``<prefix>_N`` input keys from kwargs, sorted by numeric suffix.
    Skips malformed keys (e.g. "<prefix>_" or "<prefix>_x") defensively.
    """
    marker = f"{prefix}_"
    result = []
    for k in kwargs:
        if not k.startswith(marker):
            continue
        suffix = k[len(marker):]
        if suffix.isdigit():
            result.append(k)
    return sorted(result, key=lambda k: int(k[len(marker):]))


def image_to_base64(image):
    """
    Convert a ComfyUI IMAGE tensor (BHWC, float 0-1) into a base64 PNG string.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input 'image' is not a torch.Tensor")

    if image.ndim == 4:
        if image.shape[0] != 1:
            print(f"Warning: image batch size is {image.shape[0]}, using only the first image.")
        image = image.squeeze(0)

    if image.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: {image.shape}. Expected HWC.")

    image_np = image.cpu().numpy()
    if image_np.dtype != np.uint8:
        if image_np.min() < 0 or image_np.max() > 1:
            print("Warning: image tensor values outside [0, 1]. Clamping.")
            image_np = np.clip(image_np, 0, 1)
        image_np = (image_np * 255).astype(np.uint8)

    pil_image = Image.fromarray(image_np, "RGB")
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
