from pathlib import Path

import jax

from jaxltl.deep_ltl.curriculum.curriculum import (
    MultiRandomStage,
    PrecomputedCurriculum,
    RandomCurriculumStage,
)
from jaxltl.deep_ltl.curriculum.zone_env_graph_samplers import (
    GraphZoneReachAvoidSampler,
    GraphZoneReachStaySampler,
)
from jaxltl.environments.zone_env.zone_env import ZoneEnv
from jaxltl.ltl.logic.assignment import Assignment

propositions = ZoneEnv.propositions
assignments = [Assignment(frozenset({color})) for color in propositions]
assignments.append(Assignment(frozenset()))  # empty assignment

_max_length = 3
_max_nodes = ZoneEnv.max_nodes
_max_edges = ZoneEnv.max_edges


def make(load_path: str | Path | None = None):
    return PrecomputedCurriculum(
        [
            # 1. Simple reach tasks
            RandomCurriculumStage(
                sampler=GraphZoneReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=0,
                    propositions=propositions,
                    assignments=assignments,
                    max_length=_max_length,
                    max_nodes=_max_nodes,
                    max_edges=_max_edges,
                ),
                threshold=0.9,
            ),
            # 2. Reach tasks of depth 2
            RandomCurriculumStage(
                sampler=GraphZoneReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=0,
                    propositions=propositions,
                    assignments=assignments,
                    max_length=_max_length,
                    max_nodes=_max_nodes,
                    max_edges=_max_edges,
                ),
                threshold=0.95,
            ),
            # 3. Simple reach-avoid tasks
            RandomCurriculumStage(
                sampler=GraphZoneReachAvoidSampler(
                    depth=1,
                    reach=1,
                    avoid=1,
                    propositions=propositions,
                    assignments=assignments,
                    max_length=_max_length,
                    max_nodes=_max_nodes,
                    max_edges=_max_edges,
                ),
                threshold=0.95,
            ),
            # 4. Reach-avoid tasks of depth 2
            RandomCurriculumStage(
                sampler=GraphZoneReachAvoidSampler(
                    depth=2,
                    reach=1,
                    avoid=1,
                    propositions=propositions,
                    assignments=assignments,
                    max_length=_max_length,
                    max_nodes=_max_nodes,
                    max_edges=_max_edges,
                ),
                threshold=0.9,
            ),
            # 5. Reach-avoid / reach-stay tasks
            MultiRandomStage(
                stages=[
                    RandomCurriculumStage(
                        sampler=GraphZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=GraphZoneReachStaySampler(
                            num_stay=30,
                            avoid=(0, 1),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
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
                        sampler=GraphZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=GraphZoneReachStaySampler(
                            num_stay=60,
                            avoid=(0, 1),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
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
                        sampler=GraphZoneReachAvoidSampler(
                            depth=(1, 2),
                            reach=(1, 2),
                            avoid=(0, 2),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
                        ),
                        threshold=None,
                    ),
                    RandomCurriculumStage(
                        sampler=GraphZoneReachStaySampler(
                            num_stay=60,
                            avoid=(0, 2),
                            propositions=propositions,
                            assignments=assignments,
                            max_length=_max_length,
                            max_nodes=_max_nodes,
                            max_edges=_max_edges,
                        ),
                        threshold=None,
                    ),
                ],
                probs=[0.8, 0.2],
                threshold=None,
            ),
        ],
        key=jax.random.key(0),
        num_samples=int(1e3),
        load_path=load_path,
    )
