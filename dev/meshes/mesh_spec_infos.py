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
        specs = [
            mesh.LineSpecification("line"),
            mesh.CylindricalSliceSpecification("cyl"),
            mesh.SphericalSliceSpecification("sph"),
            mesh.SphericalHarmonicSpecification("harm"),
        ]

        for s in specs:
            print(repr(s))
            print()
            print(str(s))
            print()
            print(s.info())
            print()
            print("-" * 80)
            print()
