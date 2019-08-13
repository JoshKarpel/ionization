import logging
import os

import numpy as np

import simulacra as si

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        "ionization",
        stdout_logs=True,
        stdout_level=logging.DEBUG,
        file_dir=OUT_DIR,
        file_logs=False,
    ) as logger:
        x = np.linspace(4.99, 5.01, 1e6)
        mask = ion.RadialCosineMask(inner_radius=2, outer_radius=5, smoothness=8)
        si.vis.xy_plot("mask_test", x, mask(r=x), **PLOT_KWARGS)

        # electric = ion.Rectangle(start_time = 25 * asec, end_time = 100 * asec, amplitude = 1 * atomic_electric_field)

        # mask = ion.RadialCosineMask(inner_radius = 40 * bohr_radius, outer_radius = 49 * bohr_radius)
        # sim = ion.SphericalHarmonicSpecification('mask',
        #                                          time_final = 200 * asec,
        #                                          r_bound = 50 * bohr_radius, r_points = 50 * 8,
        #                                          l_bound = 100,
        #                                          electric_potential = electric,
        #                                          test_states = [ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)],
        #                                          mask = mask).to_sim()
        #
        # sim.run_simulation()
        # sim.info().log()
        # print(sim.info())
        #
        # print(sim.mesh.norm)
        # print(sim.mesh.state_overlap(sim.spec.initial_state))
        #
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR, log = True)
