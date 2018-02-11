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
        sim = ion.mesh.SphericalHarmonicSpecification(
            'test',
            time_initial = 0 * u.asec,
            time_final = 5 * u.asec,
            time_step = 1 * u.asec,
            test_states = [ion.states.HydrogenBoundState(n, l) for n in range(1, 3) for l in range(n)],
            datastore_types = [],
        ).to_simulation()
        # sim2 = ion.mesh.SphericalHarmonicSpecification(
        #     'test2',
        #     time_initial = 0 * u.asec,
        #     time_final = 5 * u.asec,
        #     time_step = 1 * u.asec,
        #     test_states = [ion.states.HydrogenBoundState(n, l) for n in range(1, 3) for l in range(n)],
        #     initial_state = ion.states.HydrogenBoundState(2, 0),
        # ).to_simulation()

        # print(sim.spec.datastore_types)
        # print(sim.datastores)

        # print(sim.data)
        print('norm', sim.data.norm)
        # print(sim.data.inner_products)
        # print(sim.data.state_overlaps)

        # print('initial', sim.data.initial_state_overlap)
        # for k, v in sim.data.state_overlaps.items():
        #     print(k, v)

        sim.run()

        print('norm', sim.data.norm)
        # print('initial', sim.data.initial_state_overlap)
        # for k, v in sim.data.state_overlaps.items():
        #     print(k, v)

        print()

        # print('initial', sim2.data.initial_state_overlap)
        # for k, v in sim2.data.state_overlaps.items():
        #     print(k, v)
        #
        # print()

        # print('initial', sim.data.initial_state_inner_product)
        # for k, v in sim.data.inner_products.items():
        #     print(k, v)
        #
        # print(sim.info())
