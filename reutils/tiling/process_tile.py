from dataclasses import dataclass

import numpy as np
import torch

from .tile_blender import BlendDirection, TileOverlap, TileBlender
from .tiler import Tiler
from ..img_util import get_h_w_c, image2tensor, tensor2image


@dataclass
class Segment:
    start: int
    end: int
    start_padding: int
    end_padding: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def padded_length(self) -> int:
        return self.end + self.end_padding - (self.start - self.start_padding)


def split_into_segments(length: int, tile_size: int, overlap: int) -> list[Segment]:
    if length <= tile_size:
        return [Segment(0, length, 0, 0)]

    assert tile_size > overlap * 2

    result: list[Segment] = [Segment(0, tile_size - overlap, 0, overlap)]

    while result[-1].end < length:
        start_padding = overlap
        start = result[-1].end
        end = start + tile_size - overlap * 2
        end_padding = overlap

        if end + end_padding >= length:
            # Last segment
            end_padding = 0
            end = length
            start_padding = tile_size - (end - start)

        result.append(Segment(start, end, start_padding, end_padding))

    return result


def process_tiles(
    img: np.ndarray,
    model: torch.nn.Module,
    scale: int,
    tiler: Tiler,
    overlap: int = 32,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cuda'),
    amp: bool = True,
) -> np.ndarray:
    if len(img.shape) == 2:
        img = img[...,None]

    h, w, c = get_h_w_c(img)
    tile_size = tiler.starting_tile_size(w, h, c)

    model = model.to(device, dtype=dtype).eval()
    with torch.inference_mode():
        result_blender = TileBlender(h * scale, w * scale, c, BlendDirection.Y)
        y_segments = split_into_segments(h, tile_size[1], overlap)

        for y_seg in y_segments:
            y_start = y_seg.start - y_seg.start_padding
            y_end = y_seg.end + y_seg.end_padding
            row = img[y_start:y_end, :, :]

            row_blender = TileBlender((y_end - y_start) * scale, w * scale, c, BlendDirection.X)

            x_segments = split_into_segments(w, tile_size[0], overlap)
            for x_seg in x_segments:
                x_start = x_seg.start - x_seg.start_padding
                x_end = x_seg.end + x_seg.end_padding
                tile = row[:, x_start:x_end, :]
                tensor = image2tensor(tile, dtype=dtype).to(device)
                with torch.autocast(device_type=str(device), dtype=dtype, enabled=amp):
                    
                        tensor = model(tensor)

                processed_tile = tensor2image(tensor)

                x_overlap = TileOverlap(start=x_seg.start_padding * scale, end=x_seg.end_padding * scale)
                row_blender.add_tile(processed_tile, x_overlap)

            processed_row = row_blender.get_result()

            y_overlap = TileOverlap(start=y_seg.start_padding * scale, end=y_seg.end_padding * scale)
            result_blender.add_tile(processed_row, y_overlap)
            del processed_row, y_overlap

        result = result_blender.get_result()
 

    return result.squeeze()
