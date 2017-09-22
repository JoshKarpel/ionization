import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.cluster as clu

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def get_n_cycles(result):
    return result.electric_potential[0].number_of_cycles


def version_1():
    jp = clu.PulseJobProcessor.load('job_processors/hyd__pw_scan_gaussian.job')

    n_cycles = set(get_n_cycles(r) for r in jp.data.values())
    fluences = jp.parameter_set('fluence')
    phases = jp.parameter_set('phase')

    print(n_cycles, fluences, phases)


if __name__ == '__main__':
    with LOGMAN as logger:
        version_1()
