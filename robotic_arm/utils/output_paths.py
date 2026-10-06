"""Prevent accidental replacement of experiment evidence."""
from pathlib import Path


def prepare_output(path):
    path=Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True,exist_ok=True)
    return path
