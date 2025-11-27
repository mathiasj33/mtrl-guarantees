from pathlib import Path

import jax

from jaxltl.deep_ltl.curriculum.curriculum import (
    MultiRandomStage,
    PrecomputedCurriculum,
    RandomCurriculumStage,
)
from jaxltl.deep_ltl.curriculum.zone_env_samplers import (
    ZoneReachAvoidSampler,
    ZoneReachStaySampler,
)

_num_assignments = 5
_max_length = 3


def make(load_path: str | Path | None = None):
    return PrecomputedCurriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=0,
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.9,
            ),
            # 2. Reach tasks of depth 2
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=0,
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.95,
            ),
            # 3. Simple reach-avoid tasks
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
            # 4. Reach-avoid tasks of depth 2
            RandomCurriculumStage(
                sampler=ZoneReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=1,
                    num_assignments=_num_assignments,
                    max_length=_max_length,
                ),
                threshold=0.9,
            ),
            # 5. Reach-avoid / reach-stay tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=ZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=ZoneReachStaySampler(
                            num_stay=30,
                            avoid=(0, 1),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.4, 0.6],
                threshold=0.9,
            ),
            # 6. More complex reach-avoid / reach-stay tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=ZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=ZoneReachStaySampler(
                            num_stay=60,
                            avoid=(0, 1),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=0.9,
            ),
            # 7. Final mixture of complex tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=ZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=ZoneReachStaySampler(
                            num_stay=60,
                            avoid=(0, 2),
                            num_assignments=_num_assignments,
                            max_length=_max_length,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=None,
            ),
        ],
        key=jax.random.key(0),
        num_samples=int(1e6),
        load_path=load_path,
    )
