from functools import partial, reduce

import jax.numpy as jnp
import jax.random as random


def scale(
    a: jnp.ndarray,
    *,
    x_scale: float,
    y_scale: float,
    x_size: int,
    y_size: int,
):
    a_2d = jnp.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_source = x * x_scale
    y_source = y * y_scale
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = jnp.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output = output.at[i, flat_indices].set(a_2d[i, source_indices])
    return output


def rotate(
    a: jnp.ndarray,
    *,
    theta: float,
    x_size: int,
    y_size: int,
):
    a_2d = jnp.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered * jnp.cos(-theta) - y_centered * jnp.sin(-theta) + x_size / 2
    y_source = x_centered * jnp.sin(-theta) + y_centered * jnp.cos(-theta) + y_size / 2
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = jnp.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output = output.at[i, flat_indices].set(a_2d[i, source_indices])
    return output


def shear(
    a: jnp.ndarray,
    *,
    x_shear: float,
    y_shear: float,
    x_size: int,
    y_size: int,
):
    a_2d = jnp.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered + y_centered * x_shear + x_size / 2
    y_source = x_centered * y_shear + y_centered + y_size / 2
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = jnp.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output = output.at[i, flat_indices].set(a_2d[i, source_indices])
    return output


def translate(
    a: jnp.ndarray,
    *,
    x_offset: float,
    y_offset: float,
    x_size: int,
    y_size: int,
):
    a_2d = jnp.atleast_2d(a)
    batch_size = a_2d.shape[0]
    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_source = x + x_offset
    y_source = y + y_offset
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)
    valid_indices = (
        (x_source_int >= 0)
        & (x_source_int < x_size)
        & (y_source_int >= 0)
        & (y_source_int < y_size)
    )
    flat_indices = y[valid_indices] * x_size + x[valid_indices]
    source_indices = y_source_int[valid_indices] * x_size + x_source_int[valid_indices]
    output = jnp.zeros((batch_size, y_size * x_size))
    for i in range(batch_size):
        output = output.at[i, flat_indices].set(a_2d[i, source_indices])
    return output


def affine_transform(
    X: jnp.ndarray,
    x_size: int,
    y_size: int,
    *,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    min_rotation_radians: float = -jnp.pi / 4,
    max_rotation_radians: float = jnp.pi / 4,
    min_shear: float = -0.1,
    max_shear: float = 0.1,
    min_translation_pixels: int = -10,
    max_translation_pixels: int = 10,
    key: jnp.ndarray | None = None,
) -> jnp.ndarray:
    if key is None:
        key = random.PRNGKey(0)

    key1, key2, key3, key4, key5 = random.split(key, 5)

    transformations = (
        partial(
            rotate,
            theta=random.uniform(
                key1, (), minval=min_rotation_radians, maxval=max_rotation_radians
            ),
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            shear,
            x_shear=random.uniform(key2, (), minval=min_shear, maxval=max_shear),
            y_shear=random.uniform(key3, (), minval=min_shear, maxval=max_shear),
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            scale,
            x_scale=random.uniform(key4, (), minval=min_scale, maxval=max_scale),
            y_scale=random.uniform(key4, (), minval=min_scale, maxval=max_scale),
            x_size=x_size,
            y_size=y_size,
        ),
        partial(
            translate,
            x_offset=random.randint(
                key5, (), min_translation_pixels, max_translation_pixels
            ),
            y_offset=random.randint(
                key5, (), min_translation_pixels, max_translation_pixels
            ),
            x_size=x_size,
            y_size=y_size,
        ),
    )
    return reduce(lambda a, transformation: transformation(a), transformations, X)
