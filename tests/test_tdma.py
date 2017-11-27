import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np
from scipy import sparse

import ionization as ion

from . import testutils


@hyp.settings(
    deadline = None,
)
@hyp.given(
    n = st.integers(min_value = 2, max_value = 1000),
)
def test_tdma_agrees_with_solution_via_inverse(n):
    a = testutils.complex_random_sample(n - 1)
    b = testutils.complex_random_sample(n)
    c = testutils.complex_random_sample(n - 1)
    d = testutils.complex_random_sample(n)
    dia = sparse.diags([a, b, c], offsets = [-1, 0, 1])

    inv_x = np.linalg.inv(dia.toarray()).dot(d)
    tdma_x = ion.tdma(dia, d)

    np.testing.assert_allclose(dia.dot(inv_x), d)  # naive result is actually a solution
    np.testing.assert_allclose(tdma_x, inv_x)  # get same result as naive method
    np.testing.assert_allclose(dia.dot(tdma_x), d)  # tdma result is actually a solution
