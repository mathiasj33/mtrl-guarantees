import random


class ReachAvoidFormulaSampler:
    """Samples reach-avoid formulas."""

    def __init__(
        self,
        depth: int | tuple[int, int],
        reach: int | tuple[int, int],
        avoid: int | tuple[int, int],
        propositions: list[str],
    ):
        if isinstance(depth, int):
            depth = (depth, depth)
        if isinstance(reach, int):
            reach = (reach, reach)
        if isinstance(avoid, int):
            avoid = (avoid, avoid)
        self.depth = depth
        self.reach = reach
        self.avoid = avoid
        self.propositions = propositions

    def sample(self) -> str:
        depth = random.randint(self.depth[0], self.depth[1])
        props = []
        last_props = set()
        for _ in range(depth):
            nr = random.randint(self.reach[0], self.reach[1])
            na = random.randint(self.avoid[0], self.avoid[1])
            available_props = [p for p in self.propositions if p not in last_props]
            reach_props = random.sample(available_props, min(nr, len(available_props)))
            available_props = [
                p
                for p in available_props
                if p not in reach_props and p not in last_props
            ]
            avoid_props = random.sample(available_props, min(na, len(available_props)))
            props.append((reach_props, avoid_props))
            last_props = set(reach_props)
        formula = "true"
        for reach_props, avoid_props in reversed(props):
            if not avoid_props:
                formula = f"F(({' | '.join(reach_props)}) & {formula})"
            else:
                formula = f"(!({' | '.join(avoid_props)}) U ({' | '.join(reach_props)} & {formula}))"
        return formula
