"""
Implements search for paths in an LDBA that lead to accepting loops. Follows Algorithm 1
from DeepLTL (https://arxiv.org/abs/2410.04631).
"""

from dataclasses import dataclass
from typing import overload

from jaxltl.deep_ltl.reach_avoid.reach_avoid_sequence import (
    EPSILON,
    EpsilonType,
    ReachAvoidSequence,
)
from jaxltl.ltl.automata import LDBA, LDBATransition
from jaxltl.ltl.logic.assignment import Assignment


def compute_sequences(  # noqa: PLR0915
    ldba: LDBA, num_loops: int = 1
) -> dict[int, list[ReachAvoidSequence]]:
    """Computes the set of reach-avoid sequences for each LDBA state."""

    num_loops = 0 if ldba.is_finite_specification() else num_loops

    ### Helper functions ###
    def check_path_contained(path1: Path, path2: Path) -> bool:
        assert len(path2) < len(path1)
        p1 = [t[0].valid_assignments for t in path1]
        p2 = [t[0].valid_assignments for t in path2]
        acc_pos = 0
        found = False
        for p in p1:
            if p.issubset(p2[acc_pos]):
                acc_pos += 1
                if acc_pos == len(path2):
                    found = True
                    break
        return found

    def prune_paths(paths: list[Path]) -> list[Path]:
        to_remove = set()
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if i in to_remove or j in to_remove:
                    continue
                if len(paths[i]) < len(paths[j]):
                    if check_path_contained(paths[j], paths[i]):
                        to_remove.add(j)
                elif len(paths[i]) > len(paths[j]):
                    if check_path_contained(paths[i], paths[j]):
                        to_remove.add(i)
                if i in to_remove:
                    break
        paths = [paths[i] for i in range(len(paths)) if i not in to_remove]
        return paths

    ### Main DFS function ###
    def dfs(
        state: int,
        current_path: list[LDBATransition],
        state_to_path_index: dict[int, int],
        accepting_transition: LDBATransition | None,
    ) -> list[Path]:
        """
        Performs a depth-first search on the LDBA to find all simple paths leading to an accepting loop.
        """
        state_to_path_index[state] = len(current_path)
        neg_transitions = set()
        paths = []
        for transition in ldba.state_to_transitions[state]:
            scc = ldba.state_to_scc[transition.target]
            if scc.bottom and not scc.accepting:
                neg_transitions.add(transition)
            else:
                current_path.append(transition)
                stays_in_scc = scc == ldba.state_to_scc[transition.source]
                updated_accepting_transition = accepting_transition
                if transition.accepting and stays_in_scc:
                    updated_accepting_transition = transition
                if transition.target in state_to_path_index:  # found cycle
                    if (
                        updated_accepting_transition
                        in current_path[state_to_path_index[transition.target] :]
                    ):
                        # found accepting cycle
                        path = Path(
                            reach_avoid=[],
                            loop_index=state_to_path_index[transition.target],
                        )
                        future_paths = [path]
                    else:
                        # found non-accepting cycle
                        current_path.pop()
                        if transition.source != transition.target:
                            neg_transitions.add(transition)
                        continue
                else:
                    future_paths = dfs(
                        transition.target,
                        current_path,
                        state_to_path_index,
                        updated_accepting_transition,
                    )
                    if len(future_paths) == 0:
                        neg_transitions.add(transition)
                for fp in future_paths:
                    # avoid transitions can only be added once the recursion is finished, so only set() for now
                    paths.append(fp.prepend(transition, set()))
                current_path.pop()

        del state_to_path_index[state]
        paths = prune_paths(paths)
        for path in paths:
            path[0][1].update(neg_transitions)  # now we update the negative transitions
            # Note: our implementation differs from DeepLTL here, since we do not evaluate
            # the value function to decide which negative transitions to include. Instead,
            # we only include negative transitions that lead to sink states, or back along
            # the current path.
        return paths

    ### Compute sequences for each state ###
    if ldba.initial_state is None:
        raise ValueError("LDBA not initialized.")
    state_to_sequences = {}
    for state in ldba.states:
        paths = dfs(state, [], {}, None)
        state_to_sequences[state] = [path.to_sequence(num_loops) for path in paths]
    return state_to_sequences


@dataclass
class Path:
    reach_avoid: list[tuple[LDBATransition, set[LDBATransition]]]
    loop_index: int

    def __len__(self):
        return len(self.reach_avoid)

    @overload
    def __getitem__(self, item: int) -> tuple[LDBATransition, set[LDBATransition]]: ...

    @overload
    def __getitem__(
        self, item: slice
    ) -> list[tuple[LDBATransition, set[LDBATransition]]]: ...

    def __getitem__(
        self, item: int | slice
    ) -> (
        tuple[LDBATransition, set[LDBATransition]]
        | list[tuple[LDBATransition, set[LDBATransition]]]
    ):
        return self.reach_avoid[item]

    def prepend(self, reach: LDBATransition, avoid: set[LDBATransition]) -> "Path":
        return Path([(reach, avoid)] + self.reach_avoid, self.loop_index)

    def to_sequence(self, num_loops: int) -> ReachAvoidSequence:
        seq = [
            self.reach_avoid_to_assignments(r, a)
            for r, a in self.reach_avoid[: self.loop_index]
        ]
        loop = [
            self.reach_avoid_to_assignments(r, a)
            for r, a in self.reach_avoid[self.loop_index :]
        ]
        seq = seq + loop * num_loops
        return ReachAvoidSequence(seq)

    @staticmethod
    def reach_avoid_to_assignments(
        reach: LDBATransition, avoid: set[LDBATransition]
    ) -> tuple[frozenset[Assignment] | EpsilonType, frozenset[Assignment]]:
        all_avoid = [a.valid_assignments for a in avoid]
        all_avoid = set() if not all_avoid else set.union(*all_avoid)
        updated_reach = (
            EPSILON if reach.is_epsilon() else frozenset(reach.valid_assignments)
        )
        return updated_reach, frozenset(all_avoid)
