"""Script to convert LTL formulas to LDBAs and plot them using Graphviz."""

import enum

from graphviz import Source

from jaxltl.ltl.automata import LDBA, ltl2ldba
from jaxltl.ltl.logic import Assignment


class Color(enum.Enum):
    SINK = "tomato"
    ACCEPTING = "lightskyblue"
    ROOT = "#ffcc00"

    def __str__(self):
        return self.value


def draw_ldba(
    ldba: LDBA,
    filename="ldba",
    fmt="pdf",
    view=True,
    positive_label=False,
    self_loops=True,
) -> None:
    """Draw an LDBA as a graph using Graphviz."""

    dot = 'digraph "" {\n'
    dot += "rankdir=LR\n"
    dot += 'labelloc="t"\n'
    dot += 'node [shape="circle"]\n'
    dot += 'I [label="", style=invis, width=0]\n'
    dot += f"I -> {ldba.initial_state}\n"
    for state, transitions in ldba.state_to_transitions.items():
        dot += f'{state} [label="{state}" fontname="helvetica"'
        if state == ldba.sink_state:
            dot += f' color="{Color.SINK}" style="filled"'
        elif state == ldba.initial_state:
            dot += f' color="{Color.ROOT}" style="filled"'
        dot += "]\n"
        for transition in transitions:
            if not self_loops and transition.target == state:
                continue
            dot += f'{state} -> {transition.target} [label="{transition.label if not positive_label else transition.positive_label}"'
            if transition.accepting:
                dot += f' color="{Color.ACCEPTING}"'
            dot += ' fontname="helvetica"'
            dot += "]\n"
    dot += "}"
    s = Source(dot, filename=filename, format=fmt)
    s.render(view=view, cleanup=True)


def construct_ldba(formula: str, prune: bool = True) -> LDBA:
    ldba = ltl2ldba(formula)
    print("Constructed LDBA.")
    assert ldba.check_valid()
    print("Checked valid.")
    if prune:
        ldba.prune(Assignment.zero_or_one_propositions(set(ldba.propositions)))
        print("Pruned impossible transitions.")
    ldba.complete_sink_state()
    print("Added sink state.")
    ldba.compute_sccs()
    return ldba


if __name__ == "__main__":
    f = "(green U F(yellow & F(purple & F(red)))) U ((red & F(purple)) & (!yellow U green))"

    ldba = construct_ldba(f, prune=True)
    print(f"Finite: {ldba.is_finite_specification()}")
    print(f"Num states: {ldba.num_states}")
    draw_ldba(ldba, fmt="png", positive_label=True, self_loops=True)
