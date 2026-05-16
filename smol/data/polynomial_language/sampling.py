"""Sampling helpers for polynomial language data."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Sequence

from .core import Monomial, Polynomial, PolynomialDatum, Term


@dataclass(frozen=True)
class PolynomialSamplerConfig:
    """Configuration for random polynomial data point generation."""

    variables: tuple[str, ...] = ("x", "y")
    max_degree: int = 3
    num_terms: int = 4
    coefficient_min: int = -5
    coefficient_max: int = 5
    num_evaluations: int = 3
    point_min: int = -3
    point_max: int = 3
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("At least one variable is required")
        if self.max_degree < 0:
            raise ValueError("max_degree must be non-negative")
        if self.num_terms < 0:
            raise ValueError("num_terms must be non-negative")
        if self.coefficient_min > self.coefficient_max:
            raise ValueError("coefficient_min must be <= coefficient_max")
        if self.coefficient_min == 0 and self.coefficient_max == 0:
            raise ValueError("Coefficient range must include a non-zero value")
        if self.num_evaluations < 0:
            raise ValueError("num_evaluations must be non-negative")
        if self.point_min > self.point_max:
            raise ValueError("point_min must be <= point_max")


def sample_polynomial_datum(
    config: PolynomialSamplerConfig | None = None,
) -> PolynomialDatum:
    """Sample one polynomial datum with exact integer coefficients and values."""

    config = config or PolynomialSamplerConfig()
    rng = random.Random(config.seed)
    monomials = _monomials_up_to_degree(config.variables, config.max_degree)
    selected = rng.sample(monomials, k=min(config.num_terms, len(monomials)))
    terms = tuple(
        Term(
            _sample_nonzero_int(rng, config.coefficient_min, config.coefficient_max),
            monomial,
        )
        for monomial in selected
    )
    polynomial = Polynomial(terms=terms, variables=config.variables)
    points = tuple(
        _sample_point(rng, config.variables, config.point_min, config.point_max)
        for _ in range(config.num_evaluations)
    )
    return PolynomialDatum.from_points(polynomial, points)


def _monomials_up_to_degree(
    variables: Sequence[str], max_degree: int
) -> list[Monomial]:
    monomials: list[Monomial] = []
    for powers in product(range(max_degree + 1), repeat=len(variables)):
        if sum(powers) <= max_degree:
            monomials.append(Monomial(tuple(zip(variables, powers))))
    return monomials


def _sample_nonzero_int(rng: random.Random, low: int, high: int) -> int:
    value = 0
    while value == 0:
        value = rng.randint(low, high)
    return value


def _sample_point(
    rng: random.Random, variables: Sequence[str], low: int, high: int
) -> dict[str, int]:
    return {variable: rng.randint(low, high) for variable in variables}
