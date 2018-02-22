import logging
import os

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = ion.mesh.SphericalHarmonicSpecification(
            'speed_test',
            r_bound = 100 * u.bohr_radius,
            r_points = 1000,
            l_bound = 100,
            # test_states = [ion.states.HydrogenBoundState(n, l) for n in range(1, 11) for l in range(n)],
            use_numeric_eigenstates = False,
            time_initial = 0,
            time_final = 1000 * u.asec,
            time_step = 1 * u.asec,
            store_data_every = -1,
            operators = ion.mesh.SphericalHarmonicLengthGaugeOperators(),
            evolution_method = ion.mesh.SphericalHarmonicSplitOperator(),
            datastore_types = (
                ion.mesh.Fields,
                ion.mesh.Norm,
                ion.mesh.InnerProducts,
                ion.mesh.RExpectationValue,
                ion.mesh.ZExpectationValue,
                ion.mesh.InternalEnergyExpectationValue,
                ion.mesh.TotalEnergyExpectationValue,
            ),
        ).to_sim()

        # sim.info().log()
        with si.utils.BlockTimer() as timer:
            sim.run()
        # sim.info().log()

        mesh_points = sim.mesh.mesh_points
        time_points = sim.time_steps - 1
        space_time_points = mesh_points * time_points

        logger.info(f'Number of Space Points: {mesh_points}')
        logger.info(f'Number of Time Points: {time_points}')
        logger.info(f'Number of Space-Time Points: {space_time_points}')
        logger.info(f'Sim Runtime: {sim.running_time}')
        logger.info(f'BlockTimer: {timer}')

        logger.info(f'Space-Time Points / Runtime (according to Sim): {round(space_time_points / sim.running_time.total_seconds())}')
        logger.info(f'Space-Time Points / Runtime (according to BlockTimer): {round(space_time_points / timer.proc_time_elapsed)}')

        logger.info(f'Final Norm: {sim.data.norm[-1]}')
        logger.info(f'Initial State Overlap: {sim.data.initial_state_overlap[-1]}')
