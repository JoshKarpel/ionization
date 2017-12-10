#!/usr/bin/env python

from tqdm import tqdm

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import ionization as ion
import ionization.cluster as iclu
import ionization.jobutils as ju

JOB_PROCESSOR_TYPE = iclu.MeshJobProcessor

if __name__ == '__main__':
    args = ju.parse_args(description = 'Create a TDSE Ionization vs Pulse Width, Phase, and Fluence job.')

    with ju.get_log_manager(args) as logger:
        ju.check_job_dir(args)

        parameters = []

        spec_type, mesh_kwargs = ju.ask_mesh_type()
        ju.ask_mask__radial_cosine(parameters, mesh_kwargs)
        ju.ask_numeric_eigenstate_basis(parameters, spec_type = spec_type)

        ju.ask_initial_state_for_hydrogen_sim(parameters)

        ju.ask_evolution_gauge(parameters, spec_type = spec_type)
        ju.ask_evolution_method_tdse(parameters, spec_type = spec_type)

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
        ju.ask_data_storage_tdse(parameters, spec_type = spec_type)

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')
        for job_number, spec_kwargs in enumerate(tqdm(spec_kwargs_list, ascii = True)):
            electric_potential = spec_kwargs['electric_potential']

            time_initial = time_initial_in_pw * electric_potential.pulse_width
            time_final = (time_final_in_pw * electric_potential.pulse_width) + extra_time

            spec = spec_type(
                name = job_number,
                time_initial = time_initial, time_final = time_final,
                **mesh_kwargs,
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
