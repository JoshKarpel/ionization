import time

time.sleep(5)

import sys
import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    time.sleep(5)

    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ) as logger:
        R, ppR, L, T, maxE, maxL = sys.argv[1:]

        R = int(R)
        ppR = int(ppR)
        L = int(L)
        T = int(T)
        maxE = float(maxE) * u.eV
        maxL = int(maxL)

        sim = mesh.SphericalHarmonicSpecification(
            f"ram_test__R={R}_ppR={ppR}_L={L}",
            r_bound=R * u.bohr_radius,
            r_points=R * ppR,
            l_bound=L,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=maxE,
            numeric_eigenstate_max_angular_momentum=maxL,
            time_initial=0,
            time_final=T * u.asec,
            time_step=1 * u.asec,
            store_data_every=1,
            evolution_method=mesh.SplitInteractionOperator(),
        ).to_sim()
        logger.info(sim.info())

        time.sleep(5)

        sim.run()

        logger.info(sim.info())

        time.sleep(5)
