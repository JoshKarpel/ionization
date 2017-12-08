cimport cython
import numpy as np
cimport numpy as np

complex = np.complex128
ctypedef np.complex128_t complex_t

float = np.float64
ctypedef np.float64_t float_t

ctypedef np.int_t int_t

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def tdma(matrix, np.ndarray[complex_t, ndim = 1] d):
    """
    Return the result of multiplying d by the inverse of matrix.

    :param matrix: A scipy diagonal sparse matrix.
    :param d: A numpy array.
    :return: x = m^-1 d
    """
    cdef np.ndarray[complex_t, ndim = 1] subdiagonal = np.concatenate((np.zeros(1), matrix.data[0]))
    cdef np.ndarray[complex_t, ndim = 1] diagonal = matrix.data[1]
    cdef np.ndarray[complex_t, ndim = 1] superdiagonal = matrix.data[2, 1:]

    # cdef int n = len(d)  # dimension of the matrix, also the number of equations we have to solve
    cdef Py_ssize_t n = len(d)  # dimension of the matrix, also the number of equations we have to solve

    cdef np.ndarray[complex_t, ndim = 1] new_superdiagonal = np.zeros(n - 1, dtype = complex)  # allocate in advance so we don't have to keep appending
    cdef np.ndarray[complex_t, ndim = 1] new_d = np.zeros(n, dtype = complex)
    cdef np.ndarray[complex_t, ndim = 1] x = np.zeros(n, dtype = complex)

    # cdef int i
    cdef Py_ssize_t i, im
    cdef complex_t sub_i, denom
    # cdef complex_t sub_i, diag_i, new_super_im, denom

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
