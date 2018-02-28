#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        specs = [
            ion.mesh.LineSpecification('line'),
            ion.mesh.CylindricalSliceSpecification('cyl'),
            ion.mesh.SphericalSliceSpecification('sph'),
            ion.mesh.SphericalHarmonicSpecification('harm'),
        ]

        for s in specs:
            print(repr(s))
            print()
            print(str(s))
            print()
            print(s.info())
            print()
            print('-' * 80)
            print()
