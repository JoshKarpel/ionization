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
        potentials = [
            ion.potentials.ImaginaryGaussianRing(),
            ion.potentials.RadialCosineMask(),
            ion.potentials.Rectangle(window = ion.potentials.RectangularWindow()),
            ion.potentials.SineWave.from_photon_energy(13.6 * u.eV, phase = u.pi / 2, window = ion.potentials.LogisticWindow()),
            ion.potentials.SincPulse(phase = u.pi, window = ion.potentials.LinearRampWindow()),
            ion.potentials.GaussianPulse(phase = u.pi / 3, window = ion.potentials.SmoothedTrapezoidalWindow(time_front = 100 * u.asec, time_plateau = 1000 * u.asec)),
            ion.potentials.SechPulse(phase = u.pi / 3, window = ion.potentials.SmoothedTrapezoidalWindow(time_front = 100 * u.asec, time_plateau = 1000 * u.asec)),
            ion.potentials.CoulombPotential(),
            ion.potentials.SoftCoulombPotential(),
            ion.potentials.HarmonicOscillator(),
        ]

        for p in potentials:
            print(repr(p))
            print()
            print(str(p))
            print()
            print(p.info())
            print()
            print('-' * 80)
            print()
