#!/usr/bin/env python3
"""Check available JAX devices."""

import subprocess

import jax
from loguru import logger

from mo_net.device import get_platform_to_device, print_device_info


def check_cuda_availability():
    """Check if CUDA is available on the system."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines:
                if "NVIDIA-SMI" in line:
                    logger.info(f"CUDA Driver: {line.strip()}")
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    logger.info(f"CUDA Runtime: {cuda_version}")
                if "|" in line and "GeForce" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        gpu_info = parts[1].strip()
                        logger.info(f"GPU: {gpu_info}")
            return True
    except FileNotFoundError:
        logger.warning("nvidia-smi not found - CUDA drivers may not be installed")
        return False
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return False


def main():
    logger.info(f"JAX version: {jax.__version__}")

    logger.info("\nSystem GPU Information:")
    cuda_available = check_cuda_availability()

    if cuda_available:
        logger.info("✓ CUDA drivers are installed")
    else:
        logger.info("✗ CUDA drivers not available")

    logger.info("\nJAX Device Information:")
    devices = get_platform_to_device()
    for platform, device_list in devices.items():
        logger.info(f"  {platform.upper()}: {len(device_list)} device(s)")
        for device in device_list:
            logger.info(f"    - {device}")

    if cuda_available and not any("gpu" in platform for platform in devices.keys()):
        logger.warning("⚠️  CUDA is available but JAX is not using GPU")
        logger.warning("   JAX may be installed without CUDA support")

    logger.info("\nDetailed device info:")
    print_device_info()

    logger.info("\nTesting JAX operation...")
    x = jax.numpy.ones((1000, 1000))
    y = jax.numpy.dot(x, x)
    logger.info(f"Matrix multiplication result shape: {y.shape}")
    logger.info(f"Operation ran on: {y.device}")


if __name__ == "__main__":
    main()
