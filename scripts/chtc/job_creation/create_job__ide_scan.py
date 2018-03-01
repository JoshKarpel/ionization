#!/usr/bin/env python3

from tqdm import tqdm

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import ionization as ion
import ionization.cluster as iclu
import ionization.ide as ide
import ionization.jobutils as ju

import chtc_job_utils as chtc

JOB_PROCESSOR_TYPE = iclu.IDEJobProcessor

if __name__ == '__main__':
    args = ju.parse_args(description = "Create an IDE Ionization vs Pulse Width, Phase, and Fluence job.")

    with ju.get_log_manager(args) as logger:
        ju.check_job_dir(args)

        parameters = []

        spec_type = ide.IntegroDifferentialEquationSpecification
        evolution_method = ju.ask_evolution_method_ide(parameters)

        test_charge = u.electron_charge
        test_mass = u.electron_mass
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
                value = -(u.electron_charge / u.hbar) ** 2,
            ))

        kernel = ju.ask_ide_kernel(parameters)
        tunneling_model = ju.ask_ide_tunneling(parameters)

        # ju.ask_ide_tunneling(parameters)  # TODO: this

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

        job_dir = ju.create_job_files(
            args = args,
            specs = specs,
            do_checkpoints = do_checkpoints,
            parameters = parameters,
            pulse_parameters = pulse_parameters,
            job_processor_type = JOB_PROCESSOR_TYPE,
        )

        submit_string = chtc.generate_chtc_submit_string(
            args.job_name,
            len(specs),
            do_checkpoints = do_checkpoints
        )
        chtc.submit_check(submit_string)
        chtc.write_submit_file(submit_string, job_dir)
        if not args.dry:
            chtc.submit_job(ju.get_job_dir(args))
