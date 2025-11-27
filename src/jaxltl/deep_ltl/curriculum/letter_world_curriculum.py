from pathlib import Path

import jax

from jaxltl.deep_ltl.curriculum.curriculum import (
    PrecomputedCurriculum,
    RandomCurriculumStage,
)
from jaxltl.deep_ltl.curriculum.zone_env_samplers import (
    ZoneReachAvoidSampler,
)

_num_assignments = 13
_max_length = 3


def make(load_path: str | Path | None = None):
    return PrecomputedCurriculum(
        [
            # 1. Simple reach-avoid tasks
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=1,
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.95,
            ),
            # 2. Complex length 1 tasks
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=1,
                    reach=(1, 2),
                    avoid=(0, 2),
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.95,
            ),
            # 3. Complex length 2 tasks
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=2,
                    reach=(1, 2),
                    avoid=(1, 2),
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.95,
            ),
            # 4. Complex length 3 tasks
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=3,
                    reach=(1, 2),
                    avoid=(0, 3),
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=None,
            ),
        ],
        key=jax.random.key(0),
        num_samples=int(1e6),
        load_path=load_path,
    )
