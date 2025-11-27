"""Public API for the jaxltl package."""

from pathlib import Path

from jaxltl.environments.registration import make
from jaxltl.hydra_utils.utils import register_custom_resolvers

register_custom_resolvers()

_root = Path(__file__).parent.parent.parent
DATA_DIR = _root / "data"
DEPENDENCIES_DIR = _root / "dependencies"

__all__ = ["make", "DATA_DIR", "DEPENDENCIES_DIR"]
