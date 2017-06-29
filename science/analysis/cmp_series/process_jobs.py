import os

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        jp_names = [
            'hyd__sinc',
            'fsw__sinc',
            'ide__sinc__len',
            'ide__sinc__vel',
        ]

        jps = list(iclu.PulseJobProcessor.load(f'cmp__{jp_name}.job') for jp_name in jp_names)

        for jp in jps:
            print(jp)
