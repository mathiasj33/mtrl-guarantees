from pathlib import Path

_root = Path(__file__).parent.parent.parent
MODEL_DIR = _root / "models"
PAPER_DIR = _root / "paper"
RUN_DIR = _root / "runs"
DATA_DIR = _root / "data"

__all__ = ["MODEL_DIR", "PAPER_DIR", "RUN_DIR", "DATA_DIR"]
