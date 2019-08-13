#!/usr/bin/env python

import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        potentials = [
            potentials.ImaginaryGaussianRing(),
            potentials.RadialCosineMask(),
            potentials.Rectangle(window=potentials.RectangularWindow()),
            potentials.SineWave.from_photon_energy(
                13.6 * u.eV, phase=u.pi / 2, window=potentials.LogisticWindow()
            ),
            potentials.SincPulse(phase=u.pi, window=potentials.LinearRampWindow()),
            potentials.GaussianPulse(
                phase=u.pi / 3,
                window=potentials.SmoothedTrapezoidalWindow(
                    time_front=100 * u.asec, time_plateau=1000 * u.asec
                ),
            ),
            potentials.SechPulse(
                phase=u.pi / 3,
                window=potentials.SmoothedTrapezoidalWindow(
                    time_front=100 * u.asec, time_plateau=1000 * u.asec
                ),
            ),
            potentials.CoulombPotential(),
            potentials.SoftCoulombPotential(),
            potentials.HarmonicOscillator(),
        ]

        for p in potentials:
            print(repr(p))
            print()
            print(str(p))
            print()
            print(p.info())
            print()
            print("-" * 80)
            print()
