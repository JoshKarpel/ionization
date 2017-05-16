import logging
import os

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.linalg as linalg

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG) as logger:
        r = np.array(range(0, 4))
        l = np.array(range(0, 2))

        print('r\n', r)
        print('l\n', l)
        print()

        l_mesh, r_mesh = np.meshgrid(l, r, indexing = 'ij')

        print('r_mesh\n', r_mesh)
        print('l_mesh\n', l_mesh)
        print()

        total_elements = len(r) * len(l)

        # CONSTRUCT ORIGINAL MATRIX OPERATOR

        block = sparse.diags((-np.array(range(1, len(r))), np.array(range(1, len(r)))), offsets = (-1, 1))

        print('block\n', block.toarray())

        # off_diag_identity = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
        off_diag_identity = sparse.diags((np.ones(len(l)), np.ones(len(l))), offsets = (-1, 1))

        print('off_ident\n', off_diag_identity.toarray())

        big_matrix = sparse.kron(off_diag_identity, block)  # order matters

        print('big_matrix\n', big_matrix.toarray())
        print(big_matrix.format)

        big_matrix_dia = big_matrix.todia()

        print(big_matrix_dia.format)

        # CONSTRUCT DIAGONALIZER

        u_block = np.ones((len(r), len(r)))
        print('u_block\n', u_block)

        u_for_each_two_r_blocks = sparse.kron(sparse.csr_matrix(np.array([[1, 1], [1, -1]])), u_block)

        print('u_for_each_two_r_blocks\n', u_for_each_two_r_blocks.toarray())

        exp_big_matrix = linalg.expm(big_matrix_dia.toarray())

        print('exp', exp_big_matrix)
