import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with LOGMAN as logger:
        jp = ion.cluster.MeshJobProcessor('foo', 'test')

        print(jp)
        print(jp.__class__.__mro__)

        jp.summarize()
