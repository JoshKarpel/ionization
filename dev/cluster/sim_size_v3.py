import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = ion.SphericalHarmonicSpecification(
            'info_test',
            r_bound = 250 * bohr_radius,
            r_points = 250 * 8, l_bound = 500,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_angular_momentum = 20,
            numeric_eigenstate_max_energy = 50 * eV,
            time_initial = 0, time_final = 1000 * asec, time_step = 1 * asec,
            electric_potential = ion.SineWave.from_photon_energy(10 * eV, amplitude = 1 * atomic_electric_field),
            store_data_every = -1,
        ).to_sim()

        sim.file_name = 'pre'
        sim.info().log()

        print()

        # sim.run_simulation(progress_bar = True)
        # sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        print()

        sim.file_name = 'post'
        sim.info().log()

        print()

        sim.file_name = 'with'
        path_with_mesh = sim.save(target_dir = OUT_DIR, save_mesh = True, compressed = False)
        loaded_with_mesh = ion.ElectricFieldSimulation.load(path_with_mesh)
        logger.info(loaded_with_mesh.info())
        print(f'actual size on disk: {si.utils.get_file_size_as_string(path_with_mesh)}')

        print()

        sim.file_name = 'with_compressed'
        path_with_mesh = sim.save(target_dir = OUT_DIR, save_mesh = True, compressed = True)
        loaded_with_mesh = ion.ElectricFieldSimulation.load(path_with_mesh)
        logger.info(loaded_with_mesh.info())
        print(f'actual size on disk: {si.utils.get_file_size_as_string(path_with_mesh)}')

        print()

        sim.file_name = 'without'
        path_without_mesh = sim.save(target_dir = OUT_DIR, save_mesh = False, compressed = False)
        loaded_without_mesh = ion.ElectricFieldSimulation.load(path_without_mesh)
        logger.info(loaded_without_mesh.info())
        print(f'actual size on disk: {si.utils.get_file_size_as_string(path_without_mesh)}')

        print()

        sim.file_name = 'without_compressed'
        path_without_mesh = sim.save(target_dir = OUT_DIR, save_mesh = False, compressed = True)
        loaded_without_mesh = ion.ElectricFieldSimulation.load(path_without_mesh)
        logger.info(loaded_without_mesh.info())
        print(f'actual size on disk: {si.utils.get_file_size_as_string(path_without_mesh)}')
