import logging
import os

import simulacra as si
import simulacra.units as u

import ionization as ion

import hephaestus as heph

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        with heph.Tracer() as tracer:
            sim = ion.SphericalHarmonicSpecification(
                'speed_test',
                r_bound = 20 * u.bohr_radius,
                r_points = 100, l_bound = 20,
                test_states = (), use_numeric_eigenstates = False,
                time_initial = 0, time_final = 1 * u.asec, time_step = 1 * u.asec,
                store_data_every = 1,
                evolution_gauge = 'LEN',
                evolution_method = 'SO',
            ).to_sim()

            sim.run()
            # sim.run_simulation(progress_bar = True)

        report_filepath = os.path.join(OUT_DIR, 'report.txt')
        si.utils.ensure_dir_exists(report_filepath)
        with open(os.path.join(OUT_DIR, 'report.txt'), mode = 'w', encoding = 'utf-8') as f:
            f.write(tracer.report_text())
        with open(os.path.join(OUT_DIR, 'report.html'), mode = 'w', encoding = 'utf-8') as f:
            f.write(tracer.report_html())
