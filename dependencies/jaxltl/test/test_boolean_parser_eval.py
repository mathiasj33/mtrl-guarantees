from jaxltl.ltl.logic.assignment import Assignment
from jaxltl.ltl.logic.boolean_parser import Parser


def eval_expr(expr: str, true_props: set[str]) -> bool:
    return Parser(expr).parse().eval(Assignment(frozenset(true_props)))


def test_precedence_and_over_or():
    # a | (b & c)
    expr = "a | b & c"
    assert eval_expr(expr, {"a"}) is True
    assert eval_expr(expr, {"b"}) is False
    assert eval_expr(expr, {"c"}) is False
    assert eval_expr(expr, {"b", "c"}) is True


def test_not_and_parentheses():
    expr = "!a & (b | c)"
    assert eval_expr(expr, {"b"}) is True  # !a is True, (b|c) is True
    assert eval_expr(expr, {"a", "b"}) is False  # !a is False
    assert eval_expr(expr, set()) is False  # (b|c) is False


def test_implication_semantics():
    expr = "a => b"
    assert eval_expr(expr, set()) is True  # not a or b
    assert eval_expr(expr, {"a"}) is False  # a and not b
    assert eval_expr(expr, {"b"}) is True
    assert eval_expr(expr, {"a", "b"}) is True


def test_identifiers_with_underscores_and_digits():
    expr = "a_1 & b2"
    assert eval_expr(expr, {"a_1", "b2"}) is True
    assert eval_expr(expr, {"a_1"}) is False
