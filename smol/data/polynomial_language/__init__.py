"""Structured multivariable polynomial data points.

The module keeps symbolic polynomial data separate from its text rendering so
the same data point can later be lowered into images, code, LaTeX, or other
modalities.
"""

from .core import (
    Evaluation,
    EvaluationTextFormat,
    Monomial,
    Polynomial,
    PolynomialDatum,
    PolynomialTextFormat,
    Term,
)
from .sampling import PolynomialSamplerConfig, sample_polynomial_datum

__all__ = [
    "Evaluation",
    "EvaluationTextFormat",
    "Monomial",
    "Polynomial",
    "PolynomialDatum",
    "PolynomialSamplerConfig",
    "PolynomialTextFormat",
    "Term",
    "sample_polynomial_datum",
]
