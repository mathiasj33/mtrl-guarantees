from collections.abc import Iterable

from jaxltl.ltl.automata.ldba import LDBA
from jaxltl.ltl.automata.rabinizer import run_rabinizer
from jaxltl.utils import memory


@memory.cache
def ltl2ldba(
    formula: str,
    propositions: Iterable[str] | None = None,
) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    from jaxltl.ltl.hoa import HOAParser

    hoa = run_rabinizer(formula)
    return HOAParser(formula, hoa, propositions).parse_hoa()
