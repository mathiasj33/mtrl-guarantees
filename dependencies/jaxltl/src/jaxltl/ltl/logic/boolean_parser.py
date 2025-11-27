import functools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from jaxltl.ltl.logic.boolean_lexer import Lexer, Token, TokenType

if TYPE_CHECKING:
    from jaxltl.ltl.logic.assignment import Assignment


class Parser:
    def __init__(self, expression: str):
        lexer = Lexer(expression)
        self.tokens: list[Token] = lexer.lex()
        self.pos = 0
        self.current_token: Token | None = (
            self.tokens[self.pos] if self.tokens else None
        )

    def parse(self) -> "Node":
        result = self.parse_expression()
        if self.current_token is not None:
            raise SyntaxError(f"Unexpected token at the end: {self.current_token}")
        return result

    def parse_expression(self) -> "Node":
        node = self.parse_implication()
        while self.match(TokenType.OR):
            right = self.parse_implication()
            node = OrNode(node, right)
        return node

    def parse_implication(self) -> "Node":
        node = self.parse_or()
        while self.match(TokenType.IMPLIES):
            right = self.parse_or()
            node = ImplicationNode(node, right)
        return node

    def parse_or(self) -> "Node":
        node = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            node = OrNode(node, right)
        return node

    def parse_and(self) -> "Node":
        node = self.parse_primary()
        while self.match(TokenType.AND):
            right = self.parse_primary()
            node = AndNode(node, right)
        return node

    def parse_primary(self) -> "Node":
        if self.match(TokenType.NOT):
            return NotNode(self.parse_primary())
        elif self.match(TokenType.LPAREN):
            node = self.parse_expression()
            if not self.match(TokenType.RPAREN):
                raise SyntaxError("Expected ')'")
            return node
        else:
            return self.parse_variable()

    def parse_variable(self) -> "Node":
        if self.current_token and self.current_token.type == TokenType.VAR:
            node = VarNode(self.current_token.value)
            self.next_token()
            return node
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token}")

    def match(self, token_type: TokenType) -> bool:
        if self.current_token and self.current_token.type == token_type:
            self.next_token()
            return True
        return False

    def next_token(self) -> None:
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None


class Node(ABC):
    @abstractmethod
    def eval(self, assignment: "Assignment") -> bool:
        pass


class AndNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} & {self.right})"

    def eval(self, assignment: "Assignment") -> bool:
        return self.left.eval(assignment) and self.right.eval(assignment)


class MultiAndNode(Node):
    def __init__(self, operands: Sequence[Node]):
        assert len(operands) > 1, "MultiAndNode requires at least two operands."
        self.operands = operands

    def __repr__(self) -> str:
        return f"({' & '.join(map(str, self.operands))})"

    def eval(self, assignment: "Assignment") -> bool:
        return all(operand.eval(assignment) for operand in self.operands)


class OrNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"

    def eval(self, assignment: "Assignment") -> bool:
        return self.left.eval(assignment) or self.right.eval(assignment)


class MultiOrNode(Node):
    def __init__(self, operands: Sequence[Node]):
        assert len(operands) > 1, "MultiOrNode requires at least two operands."
        self.operands = operands

    def __repr__(self) -> str:
        return f"({' | '.join(map(str, self.operands))})"

    def eval(self, assignment: "Assignment") -> bool:
        return any(operand.eval(assignment) for operand in self.operands)


class NotNode(Node):
    def __init__(self, operand: Node):
        self.operand = operand

    def __repr__(self) -> str:
        return f"!({self.operand})"

    def eval(self, assignment: "Assignment") -> bool:
        return not self.operand.eval(assignment)


class VarNode(Node):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def eval(self, assignment: "Assignment") -> bool:
        return self.name in assignment.true_propositions


class ImplicationNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} -> {self.right})"

    def eval(self, assignment: "Assignment") -> bool:
        return (not self.left.eval(assignment)) or self.right.eval(assignment)


class EmptyNode(Node):
    def eval(self, assignment: "Assignment") -> bool:
        return len(assignment.true_propositions) == 0

    def __repr__(self) -> str:
        return "{" + "}"


@functools.lru_cache(maxsize=500_000)
def parse(expression: str) -> Node:
    return Parser(expression).parse()
