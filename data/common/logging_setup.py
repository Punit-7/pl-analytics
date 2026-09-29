import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file = RotatingFileHandler(
        log_dir / "pipeline.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()  # avoid duplicate lines if called twice
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file)
