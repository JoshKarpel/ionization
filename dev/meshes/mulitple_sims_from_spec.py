#!/usr/bin/env python

import logging
import os

import simulacra as si

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        spec = mesh.SphericalHarmonicSpecification("test")

        sim_one = spec.to_sim()

        print("made first sim")

        sim_two = spec.to_sim()

        print("made second sim")

        print(sim_one is sim_two)
        print(sim_one == sim_two)
        print(sim_one.uuid, sim_two.uuid)
