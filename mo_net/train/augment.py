import jax.numpy as jnp
import jax.random as random
from jax import jit

from mo_net.functions import TransformFn


def scale(
    a: jnp.ndarray,
    *,
    x_scale: float,
    y_scale: float,
    x_size: int,
    y_size: int,
):
    batch_size = a.shape[0]
    channels = a.shape[1]

    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_source = x * x_scale
    y_source = y * y_scale
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)

    x_source_int = jnp.clip(x_source_int, 0, x_size - 1)
    y_source_int = jnp.clip(y_source_int, 0, y_size - 1)

    output = jnp.zeros((batch_size, channels, y_size, x_size))
    for i in range(batch_size):
        for c in range(channels):
            img = a[i, c]
            transformed = img[y_source_int, x_source_int]
            output = output.at[i, c].set(transformed)

    return output


def rotate(
    a: jnp.ndarray,
    *,
    theta: float,
    x_size: int,
    y_size: int,
):
    batch_size = a.shape[0]
    channels = a.shape[1]

    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered * jnp.cos(-theta) - y_centered * jnp.sin(-theta) + x_size / 2
    y_source = x_centered * jnp.sin(-theta) + y_centered * jnp.cos(-theta) + y_size / 2
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)

    x_source_int = jnp.clip(x_source_int, 0, x_size - 1)
    y_source_int = jnp.clip(y_source_int, 0, y_size - 1)

    output = jnp.zeros((batch_size, channels, y_size, x_size))
    for i in range(batch_size):
        for c in range(channels):
            img = a[i, c]
            transformed = img[y_source_int, x_source_int]
            output = output.at[i, c].set(transformed)

    return output


def shear(
    a: jnp.ndarray,
    *,
    x_shear: float,
    y_shear: float,
    x_size: int,
    y_size: int,
):
    batch_size = a.shape[0]
    channels = a.shape[1]

    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_centered = x - x_size / 2
    y_centered = y - y_size / 2
    x_source = x_centered + y_centered * x_shear + x_size / 2
    y_source = x_centered * y_shear + y_centered + y_size / 2
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)

    x_source_int = jnp.clip(x_source_int, 0, x_size - 1)
    y_source_int = jnp.clip(y_source_int, 0, y_size - 1)

    output = jnp.zeros((batch_size, channels, y_size, x_size))
    for i in range(batch_size):
        for c in range(channels):
            img = a[i, c]
            transformed = img[y_source_int, x_source_int]
            output = output.at[i, c].set(transformed)

    return output


def translate(
    a: jnp.ndarray,
    *,
    x_offset: float,
    y_offset: float,
    x_size: int,
    y_size: int,
):
    batch_size = a.shape[0]
    channels = a.shape[1]

    y, x = jnp.mgrid[0:y_size, 0:x_size]
    x_source = x + x_offset
    y_source = y + y_offset
    x_source_int = jnp.round(x_source).astype(int)
    y_source_int = jnp.round(y_source).astype(int)

    x_source_int = jnp.clip(x_source_int, 0, x_size - 1)
    y_source_int = jnp.clip(y_source_int, 0, y_size - 1)

    output = jnp.zeros((batch_size, channels, y_size, x_size))
    for i in range(batch_size):
        for c in range(channels):
            img = a[i, c]
            transformed = img[y_source_int, x_source_int]
            output = output.at[i, c].set(transformed)

    return output


def affine_transform2D(
    *,
    x_size: int,
    y_size: int,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
    min_rotation_radians: float = -jnp.pi / 4,
    max_rotation_radians: float = jnp.pi / 4,
    min_shear: float = -0.1,
    max_shear: float = 0.1,
    min_translation_pixels: int = -10,
    max_translation_pixels: int = 10,
) -> TransformFn:
    @jit
    def _affine_transform_inner(
        X: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        key1, key2, key3, key4, key5, key6 = random.split(key, 6)
        rotation = random.uniform(
            key1, (), minval=min_rotation_radians, maxval=max_rotation_radians
        )
        x_shear = random.uniform(key2, (), minval=min_shear, maxval=max_shear)
        y_shear = random.uniform(key3, (), minval=min_shear, maxval=max_shear)
        scale_val = random.uniform(key4, (), minval=min_scale, maxval=max_scale)
        x_offset = random.randint(
            key5, (), min_translation_pixels, max_translation_pixels
        )
        y_offset = random.randint(
            key5, (), min_translation_pixels, max_translation_pixels
        )
        X = rotate(X, theta=rotation, x_size=x_size, y_size=y_size)
        X = shear(X, x_shear=x_shear, y_shear=y_shear, x_size=x_size, y_size=y_size)
        X = scale(X, x_scale=scale_val, y_scale=scale_val, x_size=x_size, y_size=y_size)
        X = translate(
            X, x_offset=x_offset, y_offset=y_offset, x_size=x_size, y_size=y_size
        )

        return X, key6

    return _affine_transform_inner
