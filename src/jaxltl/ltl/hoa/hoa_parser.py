import re
from collections.abc import Iterable
from typing import Any

from jaxltl.ltl.automata import LDBA


class HOAParser:
    """
    A parser for LDBAs given in the HOA format. Handles epsilon transitions as output by rabinizer.
    """

    def __init__(
        self,
        formula: str,
        hoa_text: str,
        propositions: Iterable[str] | None = None,
    ):
        self.formula = formula
        self.lines = hoa_text.split("\n")
        self.line_number = 0
        self.ldba = None
        self.propositions: set[str] | None = set(propositions) if propositions else None
        self.propositions_in_hoa: list[str] | None = None

    def parse_hoa(self) -> LDBA:
        if self.ldba is not None:
            return self.ldba
        self._parse_propositions()
        assert self.propositions is not None
        self.ldba = LDBA(
            self.propositions,
            formula=self.formula,
        )
        self._parse_header()
        self._parse_body()
        return self.ldba

    def _parse_propositions(self):
        self.propositions_in_hoa = self._find_and_parse_ap_line()
        if self.propositions is None:
            self.propositions = set(self.propositions_in_hoa)
        elif not set(self.propositions_in_hoa).issubset(self.propositions):
            raise ValueError(
                "Error parsing HOA. Found propositions in header that do not match given propositions."
            )

    def _find_and_parse_ap_line(self) -> list[str]:
        for num, line in enumerate(self.lines):
            if line.startswith("AP:"):
                return self._parse_ap_line(line.split(":")[1].strip(), num)
        raise ValueError("Error parsing HOA. Missing required header field `AP`.")

    @staticmethod
    def _parse_ap_line(value: str, line_number: int) -> list[str]:
        parts = value.split(" ")
        num_props = parts[0]
        props = parts[1:]
        if int(num_props) != len(props):
            raise ValueError(
                f"Error parsing HOA at line {line_number}. Expected {num_props} propositions."
            )
        return [p.replace('"', "") for p in props]

    def _parse_header(self):
        assert self.ldba is not None
        self._expect_line("HOA: v1")
        found_start = False
        while self._peek(error_msg='Expecting "--BODY--".') != "--BODY--":
            name, value = self._parse_header_line()
            match name:
                case "Start":
                    self.ldba.add_state(int(value), initial=True)
                    found_start = True
                case "acc-name":
                    self._expect("Buchi", value)
                case "Acceptance":
                    self._expect("1 Inf(0)", value)
                case _:
                    continue
        if not found_start:
            raise ValueError(
                f"Error parsing HOA at line {self.line_number}. Missing required header field `Start`."
            )

    def _parse_header_line(self) -> tuple[str, str]:
        line = self._consume(error_msg="Expecting header line.")
        if ":" not in line:
            raise ValueError(
                f"Error parsing HOA at line {self.line_number}. Expected a header line."
            )
        name, value = line.split(":")
        return name.strip(), value.strip()

    def _parse_body(self):
        self._expect_line("--BODY--")
        while self._peek(error_msg='Expecting "--END--".') != "--END--":
            self._parse_state()
        self._expect_line("--END--")

    def _parse_state(self):
        assert self.ldba is not None
        state_line = self._consume(error_msg="Expecting state line.")
        if not state_line.startswith("State: "):
            raise ValueError(
                f"Error parsing HOA at line {self.line_number}. Expected a state line."
            )
        state = int(state_line.split(" ")[1])
        self.ldba.add_state(state)
        while self._peek().startswith("[") or self._peek().isdigit():
            self._parse_transition(state)

    def _parse_transition(self, source: int):
        assert self.ldba is not None
        line = self._consume(error_msg="Expecting transition line.")
        label, line = self._parse_label(line)
        parts = line.split(" ")
        target = int(parts[0])
        self.ldba.add_state(target)
        accepting = False
        if len(parts) > 1:
            self._expect("{0}", parts[1])
            accepting = True
        self.ldba.add_transition(source, target, label, accepting)

    def _parse_label(self, line: str) -> tuple[str | None, str]:
        label = None
        if line.startswith("["):
            parts = line.split("]")
            label = parts[0][1:].strip()
            label = self._replace_numeric_propositions(label)
            line = parts[1].strip()
        return label, line

    def _replace_numeric_propositions(self, label: str) -> str:
        assert self.propositions_in_hoa is not None
        regexp = r"(\d+)"
        return re.sub(
            regexp,
            lambda m: self.propositions_in_hoa[int(m.group(0))],  # type: ignore
            label,
        )

    def _peek(self, error_msg: str | None = None) -> str:
        if self.line_number >= len(self.lines):
            raise ValueError(
                f"Error parsing HOA. Reached end of input.{'' if error_msg is None else f' {error_msg}'}"
            )
        return self.lines[self.line_number]

    def _consume(self, error_msg: str | None = None) -> str:
        line = self._peek(error_msg)
        self.line_number += 1
        return line

    def _expect_line(self, expected: str):
        if self._peek() != expected:
            raise ValueError(
                f"Error parsing HOA at line {self.line_number}. Expected: {expected}."
            )
        self.line_number += 1

    def _expect(self, expected: Any, actual: Any):
        if expected != actual:
            raise ValueError(
                f"Error parsing HOA at line {self.line_number}. Expected: {expected}."
            )
