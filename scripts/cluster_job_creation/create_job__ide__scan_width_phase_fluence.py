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
import ionization.integrodiff as ide
import ionization.jobutils as ju

JOB_PROCESSOR_TYPE = iclu.IDEJobProcessor

if __name__ == '__main__':
    args = ju.parse_args(description = "Create an IDE Ionization vs Pulse Width, Phase, and Fluence job.")

    with ju.get_log_manager(args) as logger:
        ju.check_job_dir(args)

        parameters = []

        spec_type = ide.IntegroDifferentialEquationSpecification
        evolution_gauge = ju.ask_evolution_gauge(parameters, spec_type = spec_type)
        evolution_method = ju.ask_evolution_method(parameters, spec_type = spec_type)

        test_charge = electron_charge
        test_mass = electron_mass
        test_energy = ion.HydrogenBoundState(1, 0).energy

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
                name = 'kernel_kwargs',
                value = {'omega_b': ion.HydrogenBoundState(1, 0).energy / hbar}
            )
        )

        if evolution_gauge == 'LEN':
            parameters.append(
                clu.Parameter(
                    name = 'kernel',
                    value = ide.hydrogen_kernel_LEN
                ))
        elif evolution_gauge == 'VEL':
            raise NotImplementedError("I haven't calculated the velocity-gauge hydrogen kernel yet...")

        ju.ask_time_step(parameters)
        if evolution_method in ['ARK4']:  # if method is adaptive
            parameters.append(clu.Parameter(name = 'time_step_minimum',
                                            value = asec * clu.ask_for_input('Minimum Time Step (in as)?', default = .01, cast_to = float)))

            parameters.append(clu.Parameter(name = 'time_step_maximum',
                                            value = asec * clu.ask_for_input('Maximum Time Step (in as)?', default = 10, cast_to = float)))

            parameters.append(clu.Parameter(name = 'error_on',
                                            value = clu.ask_for_input('Fractional Truncation Error Control on b or db/dt?', default = 'db/dt', cast_to = str)))

            parameters.append(clu.Parameter(name = 'epsilon',
                                            value = clu.ask_for_input('Fractional Truncation Error Limit?', default = 1e-6, cast_to = float)))

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
