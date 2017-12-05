import argparse
import os
import shutil
import datetime

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu
import ionization.ide as ide
import ionization.jobutils as ju

JOB_PROCESSOR_TYPE = iclu.IDEJobProcessor

if __name__ == '__main__':
    args = ju.parse_args(description = "Create an IDE Ionization vs Pulse Width, Phase, and Fluence job.")

    with ju.get_log_manager(args) as logger:
        ju.check_job_dir(args)

        parameters = []

        spec_type = ide.IntegroDifferentialEquationSpecification
        evolution_method = ju.ask_evolution_method_ide(parameters, spec_type = spec_type)

        test_charge = electron_charge
        test_mass = electron_mass
        test_energy = ion.states.HydrogenBoundState(1, 0).energy

        parameters.append(
            clu.Parameter(
                name = 'test_charge',
                value = test_charge
            ))
        parameters.append(
            clu.Parameter(
                name = 'test_mass',
                value = test_mass,
            ))
        parameters.append(
            clu.Parameter(
                name = 'test_energy',
                value = test_energy,
            ))
        parameters.append(
            clu.Parameter(
                name = 'integral_prefactor',
                value = -(electron_charge / hbar) ** 2,
            ))

        parameters.append(
            clu.Parameter(
                name = 'kernel',
                value = ide.LengthGaugeHydrogenKernel(),
            ))

        ju.ask_time_step(parameters)

        time_initial_in_pw, time_final_in_pw, extra_time = ju.ask_time_evolution_by_pulse_widths()

        # PULSE PARAMETERS
        pulse_parameters = ju.construct_pulses(
            parameters,
            time_initial_in_pw = time_initial_in_pw,
            time_final_in_pw = time_final_in_pw
        )

        # MISCELLANEOUS
        do_checkpoints = ju.ask_checkpoints(parameters)
        ju.ask_data_storage_ide(parameters, spec_type = spec_type)

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')
        for job_number, spec_kwargs in tqdm(enumerate(spec_kwargs_list)):
            electric_potential = spec_kwargs['electric_potential']

            time_initial = time_initial_in_pw * electric_potential.pulse_width
            time_final = (time_final_in_pw * electric_potential.pulse_width) + extra_time

            spec = ide.IntegroDifferentialEquationSpecification(
                name = job_number,
                time_initial = time_initial, time_final = time_final,
                **spec_kwargs,
            )

            ju.transfer_potential_attrs_to_spec(electric_potential, spec)

            spec.time_initial_in_pw = time_initial_in_pw
            spec.time_final_in_pw = time_final_in_pw

            specs.append(spec)

        ju.create_job_files(
            args = args,
            specs = specs,
            do_checkpoints = do_checkpoints,
            parameters = parameters,
            pulse_parameters = pulse_parameters,
            job_processor_type = JOB_PROCESSOR_TYPE,
        )

        if not args.dry:
            clu.submit_job(ju.get_job_dir(args))
