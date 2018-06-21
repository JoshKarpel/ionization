#!/usr/bin/env python

import os
import socket
import datetime
import argparse
import logging
from pathlib import Path

import simulacra as si


def ensure_compatibility_spec(spec):
    pass


def ensure_compatibility_sim(sim):
    pass


def find_or_init_sim(sim_name, logger):
    try:
        sim_path = Path.cwd() / f'{sim_name}.sim'
        sim = si.Simulation.load(sim_path)
        ensure_compatibility_spec(sim.spec)
        ensure_compatibility_sim(sim)
        logger.info(f'Checkpoint found at {sim_path}, recovered simulation {sim}')
        logger.info(f'Checkpoint size is {si.utils.get_file_size_as_string(sim_path)}')
    except (FileNotFoundError, EOFError):
        spec = si.Specification.load(os.path.join(os.getcwd(), f'{sim_name}.spec'))
        ensure_compatibility_spec(spec)
        sim = spec.to_sim()
        logger.info(f'Checkpoint not found, started simulation {sim}')

    return sim


def parse_args():
    parser = argparse.ArgumentParser(description = 'Run a simulation.')
    parser.add_argument(
        'sim_name',
        type = str,
        help = 'the filename of the sim to run'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logman = si.utils.LogManager(
        '__main__', 'simulacra', 'ionization',
        stdout_logs = False,
        file_logs = True,
        file_level = logging.INFO,
        file_name = args.sim_name,
        file_mode = 'a',
    )
    with logman as logger:
        try:
            logger.info(f'Landed on execute node {socket.getfqdn()} ({socket.gethostbyname(socket.gethostname())}) at {datetime.datetime.utcnow()}.')
            logger.info(f'Execute node operating system: {os.uname()}')

            dir_contents = ", ".join(str(x) for x in Path.cwd().iterdir())
            logger.info(f'Local directory contents: {dir_contents}')

            sim = find_or_init_sim(sim_name = args.sim_name, logger = logger)

            logger.info(sim.info())
            sim.run()
            logger.info(sim.info())

            sim.save()
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    main()
