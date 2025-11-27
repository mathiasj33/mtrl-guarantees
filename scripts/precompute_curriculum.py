"""Script to precompute curriculum sequences and save them to disk. Can be used
with the PrecomputedCurriculum to speed up training startup."""

import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from jaxltl import DATA_DIR, eqx_utils
from jaxltl.deep_ltl.curriculum.curriculum import PrecomputedCurriculum

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    """Precomputes and saves curriculum sequences."""
    logger.info("Instantiating curriculum to generate samples...")
    start_time = time.time()

    # Instantiate the PrecomputedCurriculum with load_path=None to force sampling.
    curriculum: PrecomputedCurriculum = hydra.utils.call(cfg.curriculum, load_path=None)

    end_time = time.time()
    logger.info(f"Sample generation finished in {end_time - start_time:.2f} seconds.")

    # Save the samples
    save_dir = Path(DATA_DIR) / cfg.env.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg.env.precomputed_curriculum_path

    logger.info(f"Saving precomputed samples to {save_path}")
    eqx_utils.save_with_treedef(save_path, curriculum.samples)
    logger.info("Save complete.")


if __name__ == "__main__":
    main()
