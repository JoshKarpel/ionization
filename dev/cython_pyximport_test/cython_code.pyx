cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
def sum(np.ndarray[double, ndim = 1] arr):
    cdef double acc = 0
    cdef int ii

    for ii in range(len(arr)):
        acc += arr[ii]

    return acc
