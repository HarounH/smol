"""Core data structures for multivariable polynomial language data."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

Number = int | Fraction


def _coerce_number(value: Number) -> Number:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return value
    raise TypeError(f"Expected int or Fraction, got {type(value).__name__}")


def _format_number(value: Number) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    return str(value)


@dataclass(frozen=True)
class PolynomialTextFormat:
    """Options for rendering a polynomial as plain text."""

    variable_style: str = "bare"
    multiplication_symbol: str = "*"
    exponent_symbol: str = "^"
    include_multiplication_by_one: bool = False
    descending_terms: bool = True

    def format_variable(self, name: str) -> str:
        if self.variable_style == "bare":
            return name
        if self.variable_style == "function":
            return f"var({name})"
        raise ValueError(f"Unknown variable_style: {self.variable_style}")


@dataclass(frozen=True)
class EvaluationTextFormat:
    """Options for rendering a polynomial data point as text."""

    polynomial_format: PolynomialTextFormat = field(
        default_factory=PolynomialTextFormat
    )
    include_question: bool = True
    include_answer: bool = True
    polynomial_label: str = "Polynomial"
    evaluate_label: str = "Evaluate at"
    values_label: str = "Values"
    point_brackets: tuple[str, str] = ("(", ")")
    assignment_symbol: str = "="
    separator: str = ", "
    value_arrow: str = "->"


@dataclass(frozen=True)
class Monomial:
    """A product of variables raised to non-negative integer powers."""

    powers: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        cleaned: list[tuple[str, int]] = []
        for variable, exponent in self.powers:
            if not variable:
                raise ValueError("Variable names must be non-empty")
            if exponent < 0:
                raise ValueError("Exponents must be non-negative")
            if exponent > 0:
                cleaned.append((variable, exponent))
        object.__setattr__(self, "powers", tuple(sorted(cleaned)))

    @classmethod
    def from_mapping(cls, powers: Mapping[str, int]) -> Monomial:
        return cls(tuple(powers.items()))

    @property
    def degree(self) -> int:
        return sum(exponent for _, exponent in self.powers)

    def evaluate(self, point: Mapping[str, Number]) -> Number:
        value: Number = 1
        for variable, exponent in self.powers:
            if variable not in point:
                raise KeyError(f"Missing value for variable {variable!r}")
            value *= point[variable] ** exponent
        return value

    def to_text(self, fmt: PolynomialTextFormat | None = None) -> str:
        fmt = fmt or PolynomialTextFormat()
        if not self.powers:
            return "1"
        parts: list[str] = []
        for variable, exponent in self.powers:
            name = fmt.format_variable(variable)
            if exponent == 1:
                parts.append(name)
            else:
                parts.append(f"{name}{fmt.exponent_symbol}{exponent}")
        return fmt.multiplication_symbol.join(parts)


@dataclass(frozen=True)
class Term:
    """A coefficient multiplied by a monomial."""

    coefficient: Number
    monomial: Monomial = field(default_factory=Monomial)

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", _coerce_number(self.coefficient))

    @property
    def degree(self) -> int:
        return self.monomial.degree

    def evaluate(self, point: Mapping[str, Number]) -> Number:
        return self.coefficient * self.monomial.evaluate(point)

    def abs_text(self, fmt: PolynomialTextFormat | None = None) -> str:
        fmt = fmt or PolynomialTextFormat()
        coefficient = abs(self.coefficient)
        monomial_text = self.monomial.to_text(fmt)
        if self.monomial.degree == 0:
            return _format_number(coefficient)
        if coefficient == 1 and not fmt.include_multiplication_by_one:
            return monomial_text
        return (
            f"{_format_number(coefficient)}{fmt.multiplication_symbol}{monomial_text}"
        )


@dataclass(frozen=True)
class Polynomial:
    """A multivariable polynomial represented as a sequence of terms."""

    terms: tuple[Term, ...]
    variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        terms = tuple(term for term in self.terms if term.coefficient != 0)
        variables = set(self.variables)
        for term in terms:
            variables.update(variable for variable, _ in term.monomial.powers)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "variables", tuple(sorted(variables)))

    @property
    def degree(self) -> int:
        if not self.terms:
            return 0
        return max(term.degree for term in self.terms)

    def evaluate(self, point: Mapping[str, Number]) -> Number:
        return sum((term.evaluate(point) for term in self.terms), start=0)

    def to_text(self, fmt: PolynomialTextFormat | None = None) -> str:
        fmt = fmt or PolynomialTextFormat()
        if not self.terms:
            return "0"

        terms = list(self.terms)
        if fmt.descending_terms:
            terms.sort(key=lambda term: (-term.degree, term.monomial.powers))

        pieces: list[str] = []
        for idx, term in enumerate(terms):
            sign = "-" if term.coefficient < 0 else "+"
            body = term.abs_text(fmt)
            if idx == 0:
                pieces.append(f"-{body}" if sign == "-" else body)
            else:
                pieces.append(f" {sign} {body}")
        return "".join(pieces)


@dataclass(frozen=True)
class Evaluation:
    """One point evaluation of a polynomial."""

    point: Mapping[str, Number]
    value: Number

    def __post_init__(self) -> None:
        point = {
            variable: _coerce_number(value) for variable, value in self.point.items()
        }
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "value", _coerce_number(self.value))

    def point_text(self, fmt: EvaluationTextFormat | None = None) -> str:
        fmt = fmt or EvaluationTextFormat()
        left, right = fmt.point_brackets
        assignments = [
            f"{variable}{fmt.assignment_symbol}{_format_number(self.point[variable])}"
            for variable in sorted(self.point)
        ]
        return f"{left}{fmt.separator.join(assignments)}{right}"

    def to_text(self, fmt: EvaluationTextFormat | None = None) -> str:
        fmt = fmt or EvaluationTextFormat()
        return f"{self.point_text(fmt)} {fmt.value_arrow} {_format_number(self.value)}"


@dataclass(frozen=True)
class PolynomialDatum:
    """A polynomial with one or more point evaluations."""

    polynomial: Polynomial
    evaluations: tuple[Evaluation, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_points(
        cls,
        polynomial: Polynomial,
        points: Sequence[Mapping[str, Number]],
        metadata: Mapping[str, str] | None = None,
    ) -> PolynomialDatum:
        evaluations = tuple(
            Evaluation(point, polynomial.evaluate(point)) for point in points
        )
        return cls(
            polynomial=polynomial, evaluations=evaluations, metadata=metadata or {}
        )

    def to_text(self, fmt: EvaluationTextFormat | None = None) -> str:
        fmt = fmt or EvaluationTextFormat()
        lines: list[str] = []
        if fmt.include_question:
            lines.append(
                f"{fmt.polynomial_label}: {self.polynomial.to_text(fmt.polynomial_format)}"
            )
            if self.evaluations:
                point_list = fmt.separator.join(
                    evaluation.point_text(fmt) for evaluation in self.evaluations
                )
                lines.append(f"{fmt.evaluate_label}: {point_list}")
        if fmt.include_answer:
            lines.append(f"{fmt.values_label}:")
            lines.extend(evaluation.to_text(fmt) for evaluation in self.evaluations)
        return "\n".join(lines)

    def to_prompt_text(self, fmt: EvaluationTextFormat | None = None) -> str:
        fmt = fmt or EvaluationTextFormat()
        prompt_fmt = EvaluationTextFormat(
            polynomial_format=fmt.polynomial_format,
            include_question=True,
            include_answer=False,
            polynomial_label=fmt.polynomial_label,
            evaluate_label=fmt.evaluate_label,
            values_label=fmt.values_label,
            point_brackets=fmt.point_brackets,
            assignment_symbol=fmt.assignment_symbol,
            separator=fmt.separator,
            value_arrow=fmt.value_arrow,
        )
        return self.to_text(prompt_fmt)

    def to_answer_text(self, fmt: EvaluationTextFormat | None = None) -> str:
        fmt = fmt or EvaluationTextFormat()
        answer_fmt = EvaluationTextFormat(
            polynomial_format=fmt.polynomial_format,
            include_question=False,
            include_answer=True,
            polynomial_label=fmt.polynomial_label,
            evaluate_label=fmt.evaluate_label,
            values_label=fmt.values_label,
            point_brackets=fmt.point_brackets,
            assignment_symbol=fmt.assignment_symbol,
            separator=fmt.separator,
            value_arrow=fmt.value_arrow,
        )
        return self.to_text(answer_fmt)

    def to_latex(self) -> str:
        raise NotImplementedError("LaTeX lowering is not implemented yet.")

    def to_image(self) -> bytes:
        raise NotImplementedError("Image lowering is not implemented yet.")

    def to_code(self) -> str:
        raise NotImplementedError("Code lowering is not implemented yet.")
