"""Logging utility."""

import logging
import sys


def setup_logger(name: str = "distill", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger that writes to stdout with a consistent format.

    Args:
        name  : Logger name.
        level : Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers if called multiple times
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
