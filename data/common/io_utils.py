from pathlib import Path


class DataValidationError(Exception):
    """Raised when downloaded or built data fails a check."""


def write_atomic(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)
