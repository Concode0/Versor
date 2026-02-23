"""Versor logging system.

Provides structured logging under the ``versor`` hierarchy.
All task/model output goes through :func:`get_logger` instead of ``print()``.

Environment variables:
    VERSOR_LOG_LEVEL  — DEBUG / INFO (default) / WARNING / ERROR
    VERSOR_LOG_FILE   — optional path; appends plain-text log lines
"""

import logging
import os
import sys

_CONFIGURED = False

# ANSI colour codes (used only when stderr is a TTY)
_COLORS = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Adds ANSI colour to level names when writing to a TTY."""

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = _COLORS.get(record.levelno, "")
            record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def _configure_once() -> None:
    """One-time lazy init of the ``versor`` root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger("versor")
    level_name = os.environ.get("VERSOR_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    # Console handler (stderr, so tqdm on stderr is unaffected)
    fmt = "%(levelname)s %(name)s: %(message)s"
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ColorFormatter(fmt, use_color=use_color))
    root.addHandler(console)

    # Optional file handler
    log_file = os.environ.get("VERSOR_LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        ))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``versor`` hierarchy.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    _configure_once()
    return logging.getLogger(f"versor.{name}")
