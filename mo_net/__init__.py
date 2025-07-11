import contextlib
import os
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Final, Literal, cast

import jax
from loguru import logger
from more_itertools import first

__version__: Final[str] = "0.0.13"

PACKAGE_DIR: Final[Path] = Path(__file__).parent.resolve()
PROJECT_ROOT_DIR: Final[Path] = PACKAGE_DIR.parent.resolve()

DeviceType = Literal["cpu", "gpu", "auto"]
DEVICE_TYPES: Final = ("cpu", "gpu", "auto")


@contextlib.contextmanager
def suppress_native_output():
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
    with suppress_native_output():
        return {
            device_type: [d for d in jax.devices() if d.platform.lower() == device_type]
            for device_type in {d.platform.lower() for d in jax.devices()}
        }


def select_device(device_type: DeviceType = "auto") -> jax.Device:
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
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def print_device_info() -> None:
    logger.info("Available JAX devices:")
    for platform_name, device_list in get_platform_to_device().items():
        for device in device_list:
            logger.info(f"  - {platform_name.upper()}: {device}")

    logger.info(f"Default device: {first(jax.devices())}")


def parse_device_arg() -> DeviceType:
    device_type: DeviceType = "auto"

    if "--device" in sys.argv:
        device_index = sys.argv.index("--device")
        if (
            device_index + 1 < len(sys.argv)
            and sys.argv[device_index + 1] in DEVICE_TYPES
        ):
            if (dev := sys.argv[device_index + 1]) in DEVICE_TYPES:
                device_type = cast(DeviceType, dev)
            else:
                raise ValueError(f"Invalid device type: {dev}.")
            del sys.argv[device_index + 1]
        del sys.argv[device_index]

    return device_type


device_type = parse_device_arg()
set_default_device(device_type)
