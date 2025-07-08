"""Device configuration for JAX - handles GPU/MPS/CPU selection."""

import contextlib
import os
import sys
from collections.abc import Collection, Mapping
from typing import Final, Literal

import jax
from loguru import logger
from more_itertools import first

DeviceType = Literal["cpu", "gpu", "auto"]
DEVICE_TYPES: Final = ("cpu", "gpu", "auto")


@contextlib.contextmanager
def suppress_native_output():
    """Suppress output from native C/C++ libraries like jax-metal"""
    og_stdout_fd = og_stderr_fd = devnull_fd = None

    try:
        og_stdout_fd, og_stderr_fd = os.dup(1), os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError as e:
        if og_stdout_fd is not None:
            os.close(og_stdout_fd)
        if og_stderr_fd is not None:
            os.close(og_stderr_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)
        raise RuntimeError(f"Failed to set up output suppression: {e}")

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        if og_stdout_fd is not None:
            with contextlib.suppress(Exception):
                os.dup2(og_stdout_fd, 1)
                os.close(og_stdout_fd)
        if og_stderr_fd is not None:
            with contextlib.suppress(Exception):
                os.dup2(og_stderr_fd, 2)
                os.close(og_stderr_fd)
        if devnull_fd is not None:
            with contextlib.suppress(Exception):
                os.close(devnull_fd)


def get_platform_to_device() -> Mapping[str, Collection[jax.Device]]:
    """Get all available JAX devices grouped by type."""
    with suppress_native_output():
        return {
            device_type: [d for d in jax.devices() if d.platform.lower() == device_type]
            for device_type in {d.platform.lower() for d in jax.devices()}
        }


def select_device(device_type: DeviceType = "auto") -> jax.Device:
    """
    Select a JAX device based on the specified type.

    Args:
        device_type: One of "cpu", "gpu", or "auto".
                    "auto" will select the best available device.

    Returns:
        The selected JAX device.
    """
    if os.environ.get("JAX_FORCE_CPU"):
        logger.info("JAX_FORCE_CPU is set, using CPU.")
        return first(jax.devices("cpu"))

    available = get_platform_to_device()

    if device_type == "auto":
        if "gpu" in available:
            device = first(available["gpu"])
            logger.info(f"Auto-selected CUDA GPU: {device}")
            return device
        elif "metal" in available:
            device = first(available["metal"])
            logger.info(f"Auto-selected Metal/MPS device: {device}")
            return device
        else:
            device = first(available.get("cpu", jax.devices()))
            logger.info(f"Auto-selected CPU: {device}")
            return device

    elif device_type == "gpu":
        if "gpu" in available:
            device = first(available["gpu"])
            logger.info(f"Selected CUDA GPU: {device}")
            return device
        else:
            raise RuntimeError("No CUDA GPU available")

    elif device_type == "cpu":
        device = first(available.get("cpu", jax.devices()))
        logger.info(f"Selected CPU: {device}")
        return device

    else:
        raise ValueError(f"Unknown device type: {device_type}")


def set_default_device(
    device_type: DeviceType = "auto", enable_metal_fallback: bool = True
) -> None:
    """
    Set the default JAX device for all operations.

    Args:
        device_type: One of "cpu", "gpu", or "auto".
                    "auto" will select the best available device.
        enable_metal_fallback: If True and a Metal device fails compatibility test,
                              automatically restart the process with CPU fallback.
                              If False, raise an exception instead of restarting.
                              Default is True for backward compatibility.
    """
    device = select_device(device_type)

    if device.platform.lower() == "metal" and not os.environ.get("JAX_FORCE_CPU"):
        try:
            import jax.numpy as jnp

            jax.config.update("jax_default_device", device)

            test_tensor = jnp.ones((2, 2))
            _ = jnp.zeros((2, 2))
            _ = test_tensor.sum()
        except Exception as e:
            if enable_metal_fallback:
                logger.info(
                    f"Metal device {device} failed compatibility test: {e}\n"
                    "Falling back to CPU due to Metal backend issues.\n"
                )
                env = os.environ.copy()
                env["JAX_FORCE_CPU"] = "1"
                os.execve(sys.executable, [sys.executable] + sys.argv, env)
            else:
                raise RuntimeError(
                    f"Metal device {device} failed compatibility test: {e}. "
                    "Set enable_metal_fallback=True to automatically fall back to CPU, "
                    "or handle the error manually."
                ) from e
    else:
        jax.config.update("jax_default_device", device)


def enable_gpu_memory_growth() -> None:
    """
    Enable GPU memory growth to prevent JAX from pre-allocating all GPU memory.
    Useful when sharing GPU with other processes.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def print_device_info() -> None:
    """Print information about available JAX devices."""

    logger.info("Available JAX devices:")
    for platform_name, device_list in get_platform_to_device().items():
        for device in device_list:
            logger.info(f"  - {platform_name.upper()}: {device}")

    logger.info(f"Default device: {first(jax.devices())}")
