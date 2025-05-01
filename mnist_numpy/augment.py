from functools import partial, reduce

import numpy as np


def scale(
    a: np.ndarray,
    *,
    x_scale: float,
    y_scale: float,
    x_size: int,
    y_size: int,
):
    a_2d = np.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = np.mgrid[0:y_size, 0:x_size]
    x_source = x * x_scale
    y_source = y * y_scale
    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = np.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output[i, flat_indices] = a_2d[i, source_indices]
    return output


def rotate(
    a: np.ndarray,
    *,
    theta: float,
    x_size: int,
    y_size: int,
):
    a_2d = np.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = np.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered * np.cos(-theta) - y_centered * np.sin(-theta) + x_size / 2
    y_source = x_centered * np.sin(-theta) + y_centered * np.cos(-theta) + y_size / 2
    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = np.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output[i, flat_indices] = a_2d[i, source_indices]
    return output


def shear(
    a: np.ndarray,
    *,
    x_shear: float,
    y_shear: float,
    x_size: int,
    y_size: int,
):
    a_2d = np.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = np.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered + y_centered * x_shear + x_size / 2
    y_source = x_centered * y_shear + y_centered + y_size / 2
    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = np.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output[i, flat_indices] = a_2d[i, source_indices]
    return output


def translate(
    a: np.ndarray,
    *,
    x_offset: float,
    y_offset: float,
    x_size: int,
    y_size: int,
):
    a_2d = np.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = np.mgrid[0:y_size, 0:x_size]
    x_source = x + x_offset
    y_source = y + y_offset
    x_source_int = np.round(x_source).astype(int)
    y_source_int = np.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = np.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output[i, flat_indices] = a_2d[i, source_indices]
    return output


def affine_transform(
    X: np.ndarray,
    x_size: int,
    y_size: int,
    *,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    min_rotation_radians: float = -np.pi / 4,
    max_rotation_radians: float = np.pi / 4,
    min_shear: float = -0.1,
    max_shear: float = 0.1,
    min_translation_pixels: int = -10,
    max_translation_pixels: int = 10,
) -> np.ndarray:
    transformations = (
        partial(
            translate,
            x_offset=np.random.randint(min_translation_pixels, max_translation_pixels),
            y_offset=np.random.randint(min_translation_pixels, max_translation_pixels),
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            rotate,
            theta=np.random.random() * (max_rotation_radians - min_rotation_radians)
            + min_rotation_radians,
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            shear,
            x_shear=np.random.random() * (max_shear - min_shear) + min_shear,
            y_shear=np.random.random() * (max_shear - min_shear) + min_shear,
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            scale,
            x_scale=np.random.random() * (max_scale - min_scale) + min_scale,
            y_scale=np.random.random() * (max_scale - min_scale) + min_scale,
            x_size=x_size,
            y_size=y_size,
        ),
    )
    return reduce(lambda a, transformation: transformation(a), transformations, X)
