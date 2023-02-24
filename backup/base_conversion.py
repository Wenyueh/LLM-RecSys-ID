import numpy as np


def identity(n):
    return n


def quaternary(n):  # maximum length 7
    assert isinstance(n, str)
    n = int(n)
    base_four = np.base_repr(n, base=4)
    return base_four


def ternary(n):  # maximum length 9
    assert isinstance(n, str)
    n = int(n)
    base_three = np.base_repr(n, base=3)
    return base_three


def binary(n):  # maximum length 14
    assert isinstance(n, str)
    n = int(n)
    base_two = np.base_repr(n, base=2)
    return base_two
