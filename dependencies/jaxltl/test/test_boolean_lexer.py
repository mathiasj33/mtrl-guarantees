from jaxltl.ltl.logic.boolean_lexer import Lexer, TokenType


def types(tokens):
    return [t.type for t in tokens]


def values(tokens):
    return [t.value for t in tokens]


def test_lex_basic_operators_and_vars():
    expr = "a & b | !c => d"
    tokens = Lexer(expr).lex()
    assert types(tokens) == [
        TokenType.VAR,
        TokenType.AND,
        TokenType.VAR,
        TokenType.OR,
        TokenType.NOT,
        TokenType.VAR,
        TokenType.IMPLIES,
        TokenType.VAR,
    ]
    assert values(tokens) == ["a", "&", "b", "|", "!", "c", "=>", "d"]


def test_lex_parentheses_and_identifiers():
    expr = "(_x1 & y2_) | (foo_bar)"
    tokens = Lexer(expr).lex()
    assert types(tokens) == [
        TokenType.LPAREN,
        TokenType.VAR,
        TokenType.AND,
        TokenType.VAR,
        TokenType.RPAREN,
        TokenType.OR,
        TokenType.LPAREN,
        TokenType.VAR,
        TokenType.RPAREN,
    ]
    assert values(tokens) == ["(", "_x1", "&", "y2_", ")", "|", "(", "foo_bar", ")"]
