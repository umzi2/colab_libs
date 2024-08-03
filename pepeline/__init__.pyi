from __future__ import annotations

import random
from enum import Enum
from typing import Optional

import numpy as np


class ImgColor(Enum):
    GRAY = 0,
    RGB = 1,
    DYNAMIC = 2


class ImgFormat(Enum):
    U8 = 0,
    F32 = 1


class TypeNoise(Enum):
    PERLIN = 0,
    SIMPLEX = 1,
    OPENSIMPLEX = 2,
    SUPERSIMPLEX = 3,
    PERLINSURFLET = 4


class CvtType(Enum):
    RGB2Gray = 0,  #NTSC
    RGB2GrayAverage = 1,
    RGB2GrayBt709 = 2,
    RGB2GrayBt2020 = 3,
    RGB2CMYK = 4,
    CMYK2RGB = 5,
    RGB2YCbCr = 6,  #bt 601
    YCbCr2RGB = 7,  #bt 601
    RGB2YCvCrBt2020 = 8,
    YCvCr2RGBBt2020 = 9,
    RGB2YCvCrBt709 = 10,
    YCvCr2RGBBt709 = 11,
    RGB2BGR = 12,
    BGR2RGB = 13,
    GRAY2RGB = 14,
    RGB2Luma = 15  #CIELAB. L only


class TypeDot(Enum):
    CIRCLE = 0,
    CROSS = 1,
    ELLIPSE = 2,
    LINE = 2,
    INVLINE = 3


def read(
        path: str,
        mode: Optional[ImgColor] = ImgColor.DYNAMIC,
        format: Optional[ImgFormat] = ImgFormat.U8
) -> np.ndarray:
    """ The function to read the image. input parameters:
    \n path -> str file path 
    \n mode -> psd dynamic format, and in other cases rgb, None = ImgColor.DYNAMIC
    \n format -> f32 0-1 img, u8 0-255, None = ImgFormat.U8"""


def read_size(path: str) -> tuple[int, int]:
    """
    Reads the dimensions (width and height) of the image at the given path.

    Arguments:
    path -- A string that holds the path to the image file.

    Returns:
    A tuple containing the width and height of the image.

    Examples:
    dimensions = read_size("path/to/image.png")
    print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")

    Errors:
    This function will raise an error if the file does not exist, the file is not an image,
    or if there is an issue reading the image dimensions.
    """


def screentone(
        array: np.ndarray,
        dot_size: int,
        angle: Optional[int] = 0,
        dot_type: Optional[TypeDot] = 7
) -> np.ndarray:
    """
    Halftone overlay function.

    Parameters:
    - array (np.ndarray): Input array representing an image with dtype np.float32 (values ranging from 0 to 1).
    - dot_size (int): Size of the screentone dots in pixels (uint).
    - angle (None | int): Optional parameter representing the rotation angle of the halftone pattern in degrees (i16).
    - dot_type (None | TypeDot): Optional parameter specifying the type of dot pattern to use.

    Returns:
    - np.ndarray: The array with the halftone overlay applied.

    This function applies a halftone pattern overlay to the input image array.
    - The input array should be 2D with dtype np.float32 and values ranging from 0 to 1.
    - 'dot_size' determines the size of the halftone dots in pixels.
    - 'angle' specifies the rotation angle of the halftone pattern in degrees. If not provided, the pattern is not rotated.
    - 'dot_type' specifies the type of dot pattern to use. If not provided, a default dot pattern is used.
    The function returns the array with the halftone overlay applied.
    """


# def halftone(
#         array: np.ndarray,
#         dot_size: int,
#         angle: Optional[int] = None,
#         dot_type: Optional[TypeDot] = None
# ) -> np.ndarray: ...

# def cmyk_shift(
#         array: np.ndarray,
#         c_bias: [int],
#         m_bias: [int],
#         y_bias: [int],
#         k_bias: [int]
# ) -> np.ndarray: ...

def best_tile(array: np.ndarray, tile_size: int) -> (int, int):
    """
Finds the top-left corner of the tile with the highest mean Laplacian intensity.
# Arguments
* `input` - 2D image array (PyReadonlyArray2<f32>).
* `tile_size` - Size of the tile in pixels.

# Returns
* `(usize, usize)` - Coordinates of the top-left corner of the best tile.
"""


def cvt_color(array: np.ndarray, cvt_type: CvtType) -> np.ndarray:
    """
    Convert the color space of an array of type np.ndarray.

    Parameters:
    - array (np.ndarray): The input array, typically representing an image, with dtype np.float32. (YCbCr only 0-1)
    - cvt_type (CvtType): The type of color space conversion to perform.

    Returns:
    - np.ndarray: The array with the converted color space.

    This function accepts an array representing an image with dtype np.float32 and converts its color space
    based on the specified conversion type (cvt_type). The supported conversion types are enumerated in CvtType.
    The function returns the array with the converted color space.
    """


def crop_cord(array: np.ndarray) -> (
        int, int, int, int):
    """returns image coordinates not equal to 0, made for cropping using the Laplace operator"""


def fast_color_level(
        array: np.ndarray,
        in_low: Optional[int] = 0,
        in_high: Optional[int] = 255,
        out_low: Optional[int] = 0,
        out_high: Optional[int] = 255,
        gamma: Optional[float] = 1.0,
) -> np.ndarray:
    """ array:np.float32
    \n in_low...out_high:uint8
    \n gamma:float32"""


def noise_generate(
        size: tuple[int, int] | tuple[int, int, int],
        type_noise: TypeNoise,
        octaves: int,
        frequency: float,
        lacunarity: float,
        seed: Optional[int] = random.randint(0,20000),
) -> np.ndarray:
    """ size: tuple 2d or 3d
    \n type_noise: TypeNoise
    \n octaves: uint
    \n frequency: float32
    \n lacunarity: float32
    \n seed: uint"""


def save(
        array: np.ndarray,
        path: str
) -> np.ndarray:
    """function to save an image, currently supports:
                  \n     f32 0-1 array
                  \n     u8 0-255 array"""
