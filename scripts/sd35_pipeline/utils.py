from PIL import Image
import numpy as np
import torch

def value_or_default(value, default_value):
    return default_value if value is None else value

# ref. https://github.com/Stability-AI/sd3.5/blob/7df9edbf5b7b541d2b6a2296f098001f91c7b04c/sd3_impls.py#L284
def decode_latent_to_preview(x0):
    """Quick RGB approximate preview of sd3 latents"""
    factors = torch.tensor(
        [
            [-0.0645, 0.0177, 0.1052],
            [0.0028, 0.0312, 0.0650],
            [0.1848, 0.0762, 0.0360],
            [0.0944, 0.0360, 0.0889],
            [0.0897, 0.0506, -0.0364],
            [-0.0020, 0.1203, 0.0284],
            [0.0855, 0.0118, 0.0283],
            [-0.0539, 0.0658, 0.1047],
            [-0.0057, 0.0116, 0.0700],
            [-0.0412, 0.0281, -0.0039],
            [0.1106, 0.1171, 0.1220],
            [-0.0248, 0.0682, -0.0481],
            [0.0815, 0.0846, 0.1207],
            [-0.0120, -0.0055, -0.0867],
            [-0.0749, -0.0634, -0.0456],
            [-0.1418, -0.1457, -0.1259],
        ],
        device="cpu",
    )
    x0 = torch.from_numpy(x0)
    latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

    latents_ubyte = (
        ((latent_image + 1) / 2)
        .clamp(0, 1)  # change scale from -1..1 to 0..1
        .mul(0xFF)  # to 0..255
        .byte()
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())
