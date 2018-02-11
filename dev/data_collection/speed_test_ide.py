import logging
import os

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO) as logger:
        sim = ide.IntegroDifferentialEquationSpecification(
            'speed_test',
            time_initial = 0, time_final = 4000 * u.asec, time_step = 1 * u.asec,
            # store_data_every = -1,
            evolution_method = ide.RungeKuttaFourMethod(),
            # kernel = ide.LengthGaugeHydrogenKernel(),
            kernel = ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
        ).to_sim()

        # sim.info().log()
        with si.utils.BlockTimer() as timer:
            sim.run_simulation()
            # sim.run_simulation(progress_bar = True)
        # sim.info().log()

        time_points = len(sim.times) - 1

        logger.info(f'Number of Time Points: {time_points}')
        logger.info(f'Sim Runtime: {sim.running_time}')
        logger.info(f'BlockTimer: {timer}')

        logger.info(f'Time Points / Runtime (according to Sim): {round(time_points / sim.running_time.total_seconds(), 2)}')
        logger.info(f'Time Points / Runtime (according to BlockTimer): {round(time_points / timer.proc_time_elapsed, 2)}')

        logger.info(f'Initial State Overlap: {sim.b2}')
