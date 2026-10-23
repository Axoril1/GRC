"""Logging configuration for repo2video."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure application logging for repo2video.

    Configures the root logger with a standardized format and sets log levels.
    When not verbose, root is set to INFO and third-party libraries (urllib3 and
    playwright) are dampened to WARNING. When verbose, root is set to DEBUG.
    Calling this function multiple times will update configuration without
    duplicating handlers.

    Args:
        verbose: If True, set level to DEBUG; otherwise, set level to INFO.
    """
    root_logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    root_logger.setLevel(level)

    # Search for an existing handler managed by repo2video
    existing_handler: logging.Handler | None = None
    for handler in root_logger.handlers:
        if getattr(handler, "_repo2video_handler", False):
            existing_handler = handler
            break

    if existing_handler is not None:
        existing_handler.setLevel(level)
    else:
        new_handler = logging.StreamHandler()
        new_handler._repo2video_handler = True  # type: ignore[attr-defined]
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        new_handler.setFormatter(formatter)
        new_handler.setLevel(level)
        root_logger.addHandler(new_handler)

    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("playwright").setLevel(logging.WARNING)
    else:
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("playwright").setLevel(logging.DEBUG)
