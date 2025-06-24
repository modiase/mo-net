#!/usr/bin/env python3
"""Check available JAX devices."""

import jax
from loguru import logger

from mo_net.device import get_available_devices, print_device_info


def main():
    logger.info(f"JAX version: {jax.__version__}\nAvailable devices:")
    devices = get_available_devices()
    for platform, device_list in devices.items():
        logger.info(f"  {platform.upper()}: {len(device_list)} device(s)")
        for device in device_list:
            logger.info(f"    - {device}")

    logger.info("\nDetailed device info:")
    print_device_info()

    logger.info("\nTesting JAX operation...")
    x = jax.numpy.ones((1000, 1000))
    y = jax.numpy.dot(x, x)
    logger.info(f"Matrix multiplication result shape: {y.shape}")
    logger.info(f"Operation ran on: {y.device}")


if __name__ == "__main__":
    main()
