# -*- coding: utf-8 -*-
# =============================================================================
# utils/logger.py — Logging setup for entire project
# =============================================================================

import io
import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a log file.

    Args:
        name     : Logger name (usually __name__ of the calling module)
        log_dir  : Directory to write log file. If None, only logs to console.

    Returns:
        logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Formatter ─────────────────────────────────────────────────────────────
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ───────────────────────────────────────────────────────
    # On Windows the default stdout uses cp1252 which can't encode emoji/arrows.
    # Wrap it in a UTF-8 TextIOWrapper so all logger output is safe.
    try:
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        # sys.stdout may not have .buffer in some environments (e.g. IDLE, pytest)
        utf8_stdout = sys.stdout

    console_handler = logging.StreamHandler(utf8_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File Handler ──────────────────────────────────────────────────────────
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File captures everything
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger
