import torch
import numpy as np
from torch import Tensor


def get_h_w_c(img: np.ndarray):
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    return h, w, c


def image2tensor(
    value: list[np.ndarray] | np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor] | Tensor:
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if len(img.shape) == 2:
            img = torch.from_numpy(img[None, ...])
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        img.unsqueeze_(0)

        if tensor.dtype != dtype:
            img = img.to(dtype)

        return img

    if isinstance(value, list):
        return [_to_tensor(i) for i in value]
    return _to_tensor(value)


def tensor2image(
    value: list[torch.Tensor] | torch.Tensor,
    dtype=np.float32,
) -> list[np.ndarray] | np.ndarray:
    def _to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.squeeze(0).detach().cpu()

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        if len(tensor.shape) == 2:
            tensor = tensor.numpy()
        else:
            tensor = tensor.permute(1, 2, 0).numpy()

        return tensor.astype(dtype)

    if isinstance(value, list):
        return [_to_ndarray(i) for i in value]
    return _to_ndarray(value)
