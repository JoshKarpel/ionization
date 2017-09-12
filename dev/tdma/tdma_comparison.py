# run as ipython tdma_comparison

import logging
import os

from tqdm import tqdm

import numpy as np
from scipy import sparse
from numba import jit

import simulacra as si
from simulacra.units import *
import ionization as ion
from ionization.cy import tdma as tdma_cy

from IPython import get_ipython

ipython = get_ipython()

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def tdma_py(matrix, d):
    subdiagonal = np.concatenate((np.zeros(1), matrix.data[0]))
    diagonal = matrix.data[1]
    superdiagonal = matrix.data[2, 1:]

    n = len(d)

    new_superdiagonal = np.zeros(n - 1, dtype = np.complex128)
    new_d = np.zeros(n, dtype = np.complex128)
    x = np.zeros(n, dtype = np.complex128)

    new_superdiagonal[0] = superdiagonal[0] / diagonal[0]
    for i in range(1, n - 1):
        new_superdiagonal[i] = superdiagonal[i] / (diagonal[i] - (subdiagonal[i] * new_superdiagonal[i - 1]))

    new_d[0] = d[0] / diagonal[0]
    for i in range(1, n):
        new_d[i] = (d[i] - (subdiagonal[i] * new_d[i - 1])) / (diagonal[i] - (subdiagonal[i] * new_superdiagonal[i - 1]))

    x[n - 1] = new_d[n - 1]
    for i in reversed(range(0, n - 1)):
        x[i] = new_d[i] - (new_superdiagonal[i] * x[i + 1])

    return x


tdma_nu = jit(tdma_py)


if __name__ == '__main__':
    with log as logger:
        n = 100_000
        a = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        c = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)
        d = np.random.rand(n) + 1j * np.random.rand(n)

        dia = sparse.diags([a, b, c], offsets = [-1, 0, 1])
        # print(dia.toarray())

        print('\n' + ('-' * 80) + '\n')

        # print('PYTHON')
        # x = tdma_py(dia, d)
        # print(np.allclose(dia.dot(x), d))
        # ipython.magic("timeit tdma_py(dia, d)")
        #
        print('DOT PRODUCT')
        x = tdma_cy(dia, d)
        ipython.magic("timeit dia.dot(x)")

        print('\n' + ('-' * 80) + '\n')

        print('CYTHON TDMA')
        x = tdma_cy(dia, d)
        print(np.allclose(dia.dot(x), d))
        ipython.magic("timeit -r 20 tdma_cy(dia, d)")

        print('\n' + ('-' * 80) + '\n')

        # print('NUMBA TDMA')
        # x = tdma_nu(dia, d)
        # print(np.allclose(dia.dot(x), d))
        # ipython.magic("timeit tdma_nu(dia, d)")
        # ipython.magic("timeit dia.dot(x)")
