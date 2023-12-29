"""
Manifesto
Q1:
suppose we have a 1D tensor (k items - kw bits total):
    for k variables,
    each with w bits
and we want an output of 2^x bits (say x=3 since int8 is easy to deal with)
    representing 1 variable.

Q2: How to enable batched operation
Q3: Identify bottlenecks and iterate:
    - quality
    - speed
"""


class ReduceV1(torch.nn.Module):
    """ the space of all mappings is massively exponential:
    at bit level, a kw to 1 mapping is a look-up table of size 2^(kw)
    There are 2^(2^(kw)) such tables or mappings

    Given a dataset, we want to find one or more mappings that could satisfy the data
    """
    def __init__(self, k: int, w: int = 32):
        super().__init__()