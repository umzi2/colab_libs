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
            tensor = torch.from_numpy(img[None, ...])
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1)

        if img.dtype == np.uint8 and dtype.is_floating_point:
            tensor = tensor.to(dtype)
            tensor.div_(255)

        tensor.unsqueeze_(0)

        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)

        return tensor

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
            img = tensor.numpy()
        else:
            img = tensor.permute(1, 2, 0).numpy()

        if tensor.dtype.is_floating_point and dtype == np.uint8:
            img = (img * 255.0).round()

        return img.astype(dtype, copy=False)

    if isinstance(value, list):
        return [_to_ndarray(i) for i in value]
    return _to_ndarray(value)
