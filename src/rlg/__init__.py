from pathlib import Path

_root = Path(__file__).parent.parent.parent
MODEL_DIR = _root / "models"

__all__ = ["MODEL_DIR"]
