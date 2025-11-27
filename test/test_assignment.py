from jaxltl.ltl.logic.assignment import Assignment


def make_assignment(names: set[str]) -> Assignment:
    return Assignment(frozenset(names))


def test_satisfies_basic_and_special_labels():
    a = make_assignment({"a"})
    b = make_assignment({"b"})
    ab = make_assignment({"a", "b"})
    empty = make_assignment(set())

    # Special labels
    assert empty.satisfies("t") is True
    assert empty.satisfies(None) is False

    # Simple variables and boolean structure
    assert a.satisfies("a") is True
    assert a.satisfies("b") is False
    assert ab.satisfies("a & b") is True
    assert a.satisfies("a | b") is True
    assert b.satisfies("a | b") is True
    assert empty.satisfies("a | b") is False
    assert a.satisfies("!b") is True
    assert a.satisfies("a => b") is False
    assert b.satisfies("a => b") is True


def test_str_len_iter_or():
    x = make_assignment({"b", "a"})
    # Sorted inside string representation
    assert str(x) == "{a, b}"
    assert len(x) == len({"a", "b"})
    assert set(iter(x)) == {"a", "b"}

    y = make_assignment({"c"})
    z = x | y
    assert isinstance(z, Assignment)
    assert set(z) == {"a", "b", "c"}


def test_zero_or_one_propositions_contents_only():
    props = {"a", "b", "c"}
    lst = Assignment.zero_or_one_propositions(props)
    # Should contain exactly singletons and empty
    expected = {frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"c"})}
    observed = {s.true_propositions for s in lst}
    assert observed == expected


def test_all_possible_assignments_size_and_contents():
    props = ("a", "b")
    lst = Assignment.all_possible_assignments(props)
    # 2^n assignments
    assert len(lst) == 2 ** len(props)
    expected = {
        frozenset(),
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"a", "b"}),
    }
    assert {a.true_propositions for a in lst} == expected
    # First half has no 'a', second half has 'a' (deterministic property of implementation)
    first_half = lst[: len(lst) // 2]
    second_half = lst[len(lst) // 2 :]
    assert all("a" not in a.true_propositions for a in first_half)
    assert all("a" in a.true_propositions for a in second_half)
