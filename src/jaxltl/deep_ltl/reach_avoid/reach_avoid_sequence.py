from jaxltl.ltl.logic import Assignment

AssignmentSet = frozenset[Assignment]


class EpsilonType:
    def __repr__(self):
        return "EPSILON"

    def __eq__(self, other):
        return isinstance(other, EpsilonType)

    def __hash__(self):
        return hash("EPSILON")


EPSILON = EpsilonType()


class ReachAvoidSequence:
    """A reach-avoid sequence consisting of sets of assignments to reach and avoid."""

    def __init__(
        self,
        reach_avoid: list[tuple[AssignmentSet | EpsilonType, AssignmentSet]],
        repeat_last: int = 0,
    ):
        """
        Params:
            reach_avoid: A list of pairs of reach and avoid assignments or epsilon.
            repeat_last: Number of times to how often the last reach-avoid pair should be repeated.
        """
        self.reach_avoid = tuple(reach_avoid)
        self.repeat_last = repeat_last

    def __hash__(self):
        return hash(self.reach_avoid)

    def __eq__(self, other):
        if not isinstance(other, ReachAvoidSequence):
            return False
        return self.reach_avoid == other.reach_avoid

    def __len__(self):
        return len(self.reach_avoid) + self.repeat_last

    def __iter__(self):
        return iter(self.reach_avoid)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start >= len(self.reach_avoid):
                if self.repeat_last <= 0:
                    return []
                return [self.reach_avoid[-1]]
            return self.reach_avoid[item]
        if item >= len(self.reach_avoid):
            if self.repeat_last <= 0:
                raise IndexError
            return self.reach_avoid[-1]
        return self.reach_avoid[item]

    def __repr__(self):
        return f"{[f'{set(r) if not isinstance(r, EpsilonType) else EPSILON} ||| {set(a)}' for r, a in self.reach_avoid]} x {self.repeat_last}"
