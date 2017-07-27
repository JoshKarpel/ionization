import os
import argparse
import logging
import datetime as dt
import socket
import platform

import simulacra as si

import ionization as ion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ensure_compatibility(spec):
    new_storage = [
        'store_radial_position_expectation_value',
        'store_electric_dipole_moment_expectation_value'
        'store_energy_expectation_value',
        'store_norm_diff_mask',
    ]
    for stor in new_storage:
        if not hasattr(spec, stor):
            setattr(spec, stor, True)

    if isinstance(spec, ion.SphericalHarmonicSpecification) and not hasattr(spec, 'hydrogen_zero_angular_momentum_correction'):
        setattr(spec, 'hydrogen_zero_angular_momentum_correction', True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run a simulation.')
    parser.add_argument('sim_name',
                        type = str,
                        help = 'the name of the sim')

    args = parser.parse_args()

    with si.utils.LogManager('__main__', 'simulacra', 'ionization',
                             stdout_logs = False,
                             file_logs = True, file_level = logging.INFO, file_name = '{}'.format(args.sim_name), file_mode = 'a') as log:
        try:
            log.info('Loaded onto execute node {} at {}.'.format(socket.getfqdn(), dt.datetime.now()))
            log.info(os.uname())
            try:
                log.info(platform.linux_distribution())
            except Exception:
                pass
            log.info('Local directory contents: {}'.format(os.listdir(os.getcwd())))

            # try to find existing checkpoint, and start from scratch if that fails
            try:
                sim_path = os.path.join(os.getcwd(), '{}.sim'.format(args.sim_name))
                sim = si.Simulation.load(sim_path)
                log.info('Checkpoint found at {}, recovered simulation {}'.format(sim_path, sim))
                log.info('Checkpoint size is {}'.format(si.utils.get_file_size_as_string(sim_path)))
            except (FileNotFoundError, EOFError):
                # sim = si.Specification.load(os.path.join(os.getcwd(), '{}.spec'.format(args.sim_name))).to_simulation()
                spec = si.Specification.load(os.path.join(os.getcwd(), '{}.spec'.format(args.sim_name)))
                ensure_compatibility(spec)
                sim = spec.to_simulation()
                log.info('Checkpoint not found, started simulation {}'.format(sim))

            # run the simulation and save it
            log.info(sim.info())
            sim.run_simulation()
            log.info(sim.info())

            sim.save()
        except Exception as e:
            log.exception(e)
            raise e
