import torch
import numpy as np
from torch import Tensor


def get_h_w_c(img: np.ndarray):
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    return h, w, c


def image2tensor(
    img: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor] | Tensor:
    if len(img.shape) == 2:
        img = torch.from_numpy(img[None, ...])
    else:
        img = torch.from_numpy(img).permute(2, 0, 1)

    img = img.unsqueeze(0)

    if img.dtype != dtype:
        img = img.to(dtype)
    return img


def tensor2image(
    tensor: list[torch.Tensor] | torch.Tensor,
    dtype=np.float32,
) -> list[np.ndarray] | np.ndarray:
    tensor = tensor.squeeze(0).detach().cpu()

    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    if len(tensor.shape) == 2:
        tensor = tensor.numpy()
    else:
        tensor = tensor.permute(1, 2, 0).numpy()

    return tensor.astype(dtype)
