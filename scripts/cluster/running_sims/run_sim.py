#!/usr/bin/env python

import os
import socket
import datetime
import argparse
import logging

import simulacra as si


def ensure_compatibility_spec(spec):
    pass


def ensure_compatibility_sim(sim):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run a simulation.')
    parser.add_argument('sim_name',
                        type = str,
                        help = 'the name of the sim')

    args = parser.parse_args()

    logman = si.utils.LogManager(
        '__main__', 'simulacra', 'ionization',
        stdout_logs = False,
        file_logs = True, file_level = logging.INFO, file_name = '{}'.format(args.sim_name), file_mode = 'a',
    )

    with logman as logger:
        try:
            logger.info(f'Loaded onto execute node {socket.getfqdn()} (IP {socket.gethostbyname(socket.gethostname())}) at {datetime.datetime.utcnow()}.')
            logger.info(f'Execute node operating system: {os.uname()}')
            logger.info(f'Local directory contents: {os.listdir(os.getcwd())}')

            try:
                sim_path = os.path.join(os.getcwd(), '{}.sim'.format(args.sim_name))
                sim = si.Simulation.load(sim_path)
                ensure_compatibility_spec(sim.spec)
                ensure_compatibility_sim(sim)
                logger.info(f'Checkpoint found at {sim_path}, recovered simulation {sim}')
                logger.info(f'Checkpoint size is {si.utils.get_file_size_as_string(sim_path)}')
            except (FileNotFoundError, EOFError):
                spec = si.Specification.load(os.path.join(os.getcwd(), f'{args.sim_name}.spec'))
                ensure_compatibility_spec(spec)
                sim = spec.to_sim()
                logger.info(f'Checkpoint not found, started simulation {sim}')

            # run the simulation and save it
            logger.info(sim.info())
            sim.run_simulation()
            logger.info(sim.info())

            sim.save()
        except Exception as e:
            logger.exception(e)
            raise e
