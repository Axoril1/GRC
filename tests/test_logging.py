"""Tests for logging setup in repo2video."""

import logging

from repo2video.logging_setup import setup_logging


def test_setup_logging_standard() -> None:
    """Verify setup_logging configures root logger to INFO and dampens third-party loggers."""
    setup_logging(verbose=False)
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("playwright").level == logging.WARNING


def test_setup_logging_verbose() -> None:
    """Verify setup_logging with verbose=True sets DEBUG level."""
    setup_logging(verbose=True)
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert logging.getLogger("urllib3").level == logging.DEBUG
    assert logging.getLogger("playwright").level == logging.DEBUG


def test_setup_logging_idempotent() -> None:
    """Verify calling setup_logging multiple times does not duplicate handlers."""
    root_logger = logging.getLogger()

    setup_logging(verbose=False)
    handlers_count_first = len(
        [h for h in root_logger.handlers if getattr(h, "_repo2video_handler", False)]
    )

    setup_logging(verbose=False)
    setup_logging(verbose=True)
    handlers_count_after = len(
        [h for h in root_logger.handlers if getattr(h, "_repo2video_handler", False)]
    )

    assert handlers_count_first == 1
    assert handlers_count_after == 1
