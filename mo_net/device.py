"""Device configuration for JAX - handles GPU/MPS/CPU selection."""

import os
from typing import Literal

import jax
from loguru import logger

DeviceType = Literal["cpu", "gpu", "mps", "auto"]


def get_available_devices() -> dict[str, list[jax.Device]]:
    """Get all available JAX devices grouped by type."""
    devices = jax.devices()
    device_map = {}

    for device in devices:
        device_type = device.platform.lower()
        if device_type not in device_map:
            device_map[device_type] = []
        device_map[device_type].append(device)

    return device_map


def select_device(device_type: DeviceType = "auto") -> jax.Device:
    """
    Select a JAX device based on the specified type.

    Args:
        device_type: One of "cpu", "gpu", "mps", or "auto".
                    "auto" will select the best available device.

    Returns:
        The selected JAX device.
    """
    available = get_available_devices()

    if device_type == "auto":
        # Priority order: CUDA GPU > Metal/MPS > CPU
        if "gpu" in available:
            device = available["gpu"][0]
            logger.info(f"Auto-selected CUDA GPU: {device}")
            return device
        elif "metal" in available:
            device = available["metal"][0]
            logger.info(f"Auto-selected Metal/MPS device: {device}")
            return device
        else:
            device = available.get("cpu", jax.devices())[0]
            logger.info(f"Auto-selected CPU: {device}")
            return device

    elif device_type == "gpu":
        if "gpu" in available:
            device = available["gpu"][0]
            logger.info(f"Selected CUDA GPU: {device}")
            return device
        else:
            raise RuntimeError("No CUDA GPU available")

    elif device_type == "mps":
        if "metal" in available:
            device = available["metal"][0]
            logger.info(f"Selected Metal/MPS device: {device}")
            return device
        else:
            raise RuntimeError("No Metal/MPS device available")

    elif device_type == "cpu":
        device = available.get("cpu", jax.devices())[0]
        logger.info(f"Selected CPU: {device}")
        return device

    else:
        raise ValueError(f"Unknown device type: {device_type}")


def set_default_device(device_type: DeviceType = "auto") -> None:
    """
    Set the default JAX device for all operations.

    This should be called at the beginning of your program.
    """
    device = select_device(device_type)

    # Test if the device works properly with basic tensor operations
    # This is particularly important for Metal backend which has known issues
    if device.platform.lower() == "metal":
        try:
            # Test basic tensor creation that fails with Metal backend
            import jax.numpy as jnp

            # First try to set the device without testing to avoid nested context issues
            jax.config.update("jax_default_device", device)

            # Now test basic operations
            test_tensor = jnp.ones((2, 2))
            _ = jnp.zeros((2, 2))
            # Force computation to actually happen
            _ = test_tensor.sum()

            logger.info(f"Metal device {device} passed compatibility test")
        except Exception as e:
            error_msg = str(e).lower()
            if "default_memory_space" in error_msg or "unimplemented" in error_msg:
                logger.warning(
                    f"Metal device {device} has known compatibility issues: {e}"
                )
                logger.info(
                    "This is a known JAX Metal backend limitation - falling back to CPU"
                )
            else:
                logger.warning(f"Metal device {device} failed compatibility test: {e}")
                logger.info("Falling back to CPU due to Metal backend issues")

            # Fall back to CPU
            device = select_device("cpu")
            jax.config.update("jax_default_device", device)
    else:
        # For non-Metal devices, just set directly
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
    devices = get_available_devices()

    logger.info("Available JAX devices:")
    for platform, device_list in devices.items():
        for device in device_list:
            logger.info(f"  - {platform.upper()}: {device}")

    logger.info(f"Default device: {jax.devices()[0]}")


# For Apple Silicon Macs, we need to enable the Metal plugin
def setup_metal_plugin() -> None:
    """Setup Metal plugin for Apple Silicon Macs."""
    try:
        # This will be available if jax-metal is installed
        import jax_metal  # noqa: F401

        logger.info("JAX Metal plugin loaded successfully")
    except ImportError:
        logger.debug("JAX Metal plugin not available")


# Initialize Metal plugin if available
setup_metal_plugin()
