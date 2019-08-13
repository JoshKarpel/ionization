cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def tdma(matrix, np.ndarray[np.complex128_t, ndim = 1] d):
    """
    Return the result of multiplying d by the inverse of matrix.

    :param matrix: A scipy diagonal sparse matrix.
    :param d: A numpy array.
    :return: x = m^-1 d
    """
    cdef Py_ssize_t n = len(d)

    cdef np.ndarray[np.complex128_t, ndim = 1] subdiagonal = np.zeros(n + 1, dtype = np.complex128)
    subdiagonal[1:] = matrix.data[0]  # ensures proper alignment w/o needing extra index manipulation
    cdef np.ndarray[np.complex128_t, ndim = 1] diagonal = matrix.data[1]
    cdef np.ndarray[np.complex128_t, ndim = 1] superdiagonal = matrix.data[2, 1:]  # alignment

    cdef np.ndarray[np.complex128_t, ndim = 1] new_superdiagonal = np.zeros(n - 1, dtype = np.complex128)
    cdef np.ndarray[np.complex128_t, ndim = 1] new_d = np.zeros(n, dtype = np.complex128)
    cdef np.ndarray[np.complex128_t, ndim = 1] x = np.zeros(n, dtype = np.complex128)

    cdef Py_ssize_t i, im
    cdef np.complex128_t sub_i, denom

    # compute the primed superdiagonal
    new_superdiagonal[0] = superdiagonal[0] / diagonal[0]
    new_d[0] = d[0] / diagonal[0]
    for i in range(1, n - 1):
        im = i - 1

        sub_i = subdiagonal[i]
        denom = (diagonal[i] - (sub_i * new_superdiagonal[im]))

        new_superdiagonal[i] = superdiagonal[i] / denom
        new_d[i] = (d[i] - (sub_i * new_d[im])) / denom

    new_d[n - 1] = (d[n - 1] - (subdiagonal[n - 1] * new_d[n - 2])) / (diagonal[n - 1] - (subdiagonal[n - 1] * new_superdiagonal[n - 2]))

    # compute the answer
    x[n - 1] = new_d[n - 1]
    for i in range(n - 2, -1, -1):  # iterate in reversed order, since we need to construct x from back to front
        x[i] = new_d[i] - (new_superdiagonal[i] * x[i + 1])

    return x
