# smol

Ideas:
1. Build a polynomial expression language, see what can and can not fit

## Polynomial language

`smol.data.polynomial_language` provides structured synthetic examples made of
multivariable polynomials plus exact point evaluations. The data stays symbolic
until it is lowered into text, leaving room for future image, code, and LaTeX
renderers.

```python
from smol.data.polynomial_language import PolynomialSamplerConfig, sample_polynomial_datum

datum = sample_polynomial_datum(PolynomialSamplerConfig(seed=0))
print(datum.to_text())
```
