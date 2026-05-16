from smol.data.polynomial_language import (
    EvaluationTextFormat,
    Monomial,
    Polynomial,
    PolynomialSamplerConfig,
    PolynomialTextFormat,
    Term,
    sample_polynomial_datum,
)


def test_polynomial_evaluation_and_text_rendering() -> None:
    polynomial = Polynomial(
        terms=(
            Term(3, Monomial.from_mapping({"x": 2, "y": 1})),
            Term(-2, Monomial.from_mapping({"y": 1})),
            Term(5),
        ),
    )

    assert polynomial.evaluate({"x": 2, "y": 3}) == 35
    assert polynomial.to_text() == "3*x^2*y - 2*y + 5"


def test_datum_text_can_include_prompt_without_answer() -> None:
    datum = sample_polynomial_datum(
        PolynomialSamplerConfig(
            variables=("x", "y"),
            max_degree=2,
            num_terms=2,
            num_evaluations=1,
            seed=7,
        )
    )

    text = datum.to_text(EvaluationTextFormat(include_answer=False))

    assert text.startswith("Polynomial:")
    assert "Evaluate at:" in text
    assert "Values:" not in text


def test_text_format_controls_variable_and_multiplication_style() -> None:
    polynomial = Polynomial(terms=(Term(1, Monomial.from_mapping({"z": 3})),))
    fmt = PolynomialTextFormat(
        variable_style="function",
        multiplication_symbol=" ",
        include_multiplication_by_one=True,
    )

    assert polynomial.to_text(fmt) == "1 var(z)^3"


def test_datum_exposes_prompt_and_answer_text() -> None:
    datum = sample_polynomial_datum(PolynomialSamplerConfig(seed=11, num_evaluations=1))

    assert "Values:" not in datum.to_prompt_text()
    assert datum.to_answer_text().startswith("Values:")
