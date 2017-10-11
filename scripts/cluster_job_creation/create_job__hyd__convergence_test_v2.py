import argparse
import os
import shutil
import datetime

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu


if __name__ == '__main__':
    # get command line arguments
    parser = argparse.ArgumentParser(description = 'Create an Ionization vs Pulse Width, Phase, and Fluence job.')
    parser.add_argument('job_name',
                        type = str,
                        help = 'the name of the job')
    parser.add_argument('--dir', '-d',
                        action = 'store', default = os.getcwd(),
                        help = 'directory to put the job directory in. Defaults to cwd')
    parser.add_argument('--overwrite', '-o',
                        action = 'store_true',
                        help = 'force overwrite existing job directory if there is a name collision')
    parser.add_argument('--verbosity', '-v',
                        action = 'count', default = 0,
                        help = 'set verbosity level')
    parser.add_argument('--dry',
                        action = 'store_true',
                        help = 'do not attempt to actually submit the job')

    args = parser.parse_args()

    with si.utils.LogManager('simulacra', 'ionization', stdout_level = 31 - ((args.verbosity + 1) * 10)) as logger:
        # job type options
        job_processor = iclu.ConvergenceJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()
            else:
                shutil.rmtree(job_dir)

        parameters = []

        # get input from the user to define the job
        spec_type = ion.SphericalHarmonicSpecification

        r_bound = clu.Parameter(name = 'r_bound',
                                value = bohr_radius * si.cluster.ask_for_input('R Bound (Bohr radii)?', default = 50, cast_to = float))
        parameters.append(r_bound)

        parameters.append(clu.Parameter(name = 'delta_r',
                                        value = bohr_radius * np.array(clu.ask_for_eval('Radial Mesh Spacings (in Bohr radii)?',
                                                                                        default = 'np.geomspace(.1, 1, 10)')),
                                        expandable = True))

        parameters.append(clu.Parameter(name = 'l_bound',
                                        value = clu.ask_for_eval('L Bound?',
                                                                 default = 'range(100, 501, 25)'),
                                        expandable = True))

        parameters.append(clu.Parameter(name = 'time_step',
                                        value = asec * np.array(clu.ask_for_eval('Time Steps (in asec)?', default = 'np.geomspace(5, .1, 20)')),
                                        expandable = True))

        time_initial_in_pw = clu.Parameter(name = 'initial_time_in_pw',
                                           value = clu.ask_for_input('Initial Time (in pulse widths)?', default = -35, cast_to = float))
        parameters.append(time_initial_in_pw)

        parameters.append(clu.Parameter(name = 'final_time_in_pw',
                                        value = clu.ask_for_input('Final Time (in pulse widths)?', default = 40, cast_to = float)))

        initial_state = clu.Parameter(name = 'initial_state',
                                      value = ion.HydrogenBoundState(clu.ask_for_input('Initial State n?', default = 1, cast_to = int),
                                                                     clu.ask_for_input('Initial State l?', default = 0, cast_to = int)))
        parameters.append(initial_state)

        numeric_basis_q = False
        if spec_type == ion.SphericalHarmonicSpecification:
            numeric_basis_q = clu.ask_for_bool('Use numeric eigenstate basis?', default = True)
            if numeric_basis_q:
                parameters.append(clu.Parameter(name = 'use_numeric_eigenstates',
                                                value = True))
                parameters.append(clu.Parameter(name = 'numeric_eigenstate_max_angular_momentum',
                                                value = clu.ask_for_input('Numeric Eigenstate Maximum l?', default = 10, cast_to = int)))
                parameters.append(clu.Parameter(name = 'numeric_eigenstate_max_energy',
                                                value = eV * clu.ask_for_input('Numeric Eigenstate Max Energy (in eV)?', default = 50, cast_to = float)))

        if not numeric_basis_q:
            if clu.ask_for_bool('Overlap only with initial state?', default = 'yes'):
                parameters.append(clu.Parameter(name = 'test_states',
                                                value = [initial_state.value]))
            else:
                largest_n = clu.ask_for_input('Largest Bound State n to Overlap With?', default = 5, cast_to = int)
                parameters.append(clu.Parameter(name = 'test_states',
                                                value = tuple(ion.HydrogenBoundState(n, l) for n in range(largest_n + 1) for l in range(n))))

        outer_radius_default = uround(r_bound.value, bohr_radius, 2)
        parameters.append(clu.Parameter(name = 'mask',
                                        value = ion.RadialCosineMask(inner_radius = bohr_radius * clu.ask_for_input('Mask Inner Radius (in Bohr radii)?', default = outer_radius_default * .8, cast_to = float),
                                                                     outer_radius = bohr_radius * clu.ask_for_input('Mask Outer Radius (in Bohr radii)?', default = outer_radius_default, cast_to = float),
                                                                     smoothness = clu.ask_for_input('Mask Smoothness?', default = 8, cast_to = int))))

        parameters.append(clu.Parameter(name = 'evolution_gauge',
                                        value = clu.ask_for_input('Evolution Gauge? [LEN/VEL]', default = 'LEN')))

        # PULSE PARAMETERS
        pulse_parameters = []

        pulse_type_q = clu.ask_for_input('Pulse Type? [sinc/gaussian/sech]', default = 'sinc')
        pulse_names_to_types = {
            'sinc': ion.SincPulse,
            'gaussian': ion.GaussianPulse,
            'sech': ion.SechPulse,
        }
        pulse_type = pulse_names_to_types[pulse_type_q]

        pulse_width = clu.Parameter(name = 'pulse_width',
                                    value = asec * np.array(clu.ask_for_eval('Pulse Widths (in as)?', default = '[50, 100, 200, 400, 800]')),
                                    expandable = True)
        pulse_parameters.append(pulse_width)

        fluence = clu.Parameter(name = 'fluence',
                                value = (J / (cm ** 2)) * np.array(clu.ask_for_eval('Pulse Fluence (in J/cm^2)?', default = '[.01, .1, 1, 10, 20]')),
                                expandable = True)
        pulse_parameters.append(fluence)

        phases = clu.Parameter(name = 'phase',
                               value = np.array(clu.ask_for_eval('Pulse CEP (in rad)?', default = 'np.linspace(0, pi, 100)')),
                               expandable = True)
        pulse_parameters.append(phases)

        window_time_in_pw = clu.Parameter(name = 'window_time_in_pw',
                                          value = clu.ask_for_input('Window Time (in pulse widths)?', default = np.abs(time_initial_in_pw.value) - 5, cast_to = float))
        window_width_in_pw = clu.Parameter(name = 'window_width_in_pw',
                                           value = clu.ask_for_input('Window Width (in pulse widths)?', default = 0.2, cast_to = float))
        parameters.append(window_time_in_pw)
        parameters.append(window_width_in_pw)

        omega_min = clu.Parameter(name = 'omega_min',
                                  value = twopi * THz * np.array(clu.ask_for_eval('Pulse Frequency Minimum? (in THz)',
                                                                                  default = '[30]')),
                                  expandable = True)
        pulse_parameters.append(omega_min)

        pulses = tuple(pulse_type.from_omega_min(**d,
                                                 window = ion.SymmetricExponentialTimeWindow(window_time = d['pulse_width'] * window_time_in_pw.value,
                                                                                             window_width = d['pulse_width'] * window_width_in_pw.value))
                       for d in clu.expand_parameters_to_dicts(pulse_parameters))

        parameters.append(clu.Parameter(name = 'electric_potential',
                                        value = pulses,
                                        expandable = True))

        # MISCELLANEOUS
        parameters.append(clu.Parameter(name = 'electric_potential_dc_correction',
                                        value = clu.ask_for_bool('Perform Electric Field DC Correction?', default = True)))

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            time_between_checkpoints = clu.ask_for_input('How long between checkpoints (in minutes)?', default = 60, cast_to = int)
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = datetime.timedelta(minutes = time_between_checkpoints)))

        parameters.append(clu.Parameter(name = 'store_data_every',
                                        value = clu.ask_for_input('Store Data Every?', default = 1, cast_to = int)))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')

        for ii, spec_kwargs in enumerate(spec_kwargs_list):
            delta_r = spec_kwargs['delta_r']
            r_points = int(r_bound.value / delta_r)

            time_step = spec_kwargs['time_step']
            electric_potential = spec_kwargs['electric_potential']

            time_initial = spec_kwargs['initial_time_in_pw'] * electric_potential.pulse_width
            time_final = spec_kwargs['final_time_in_pw'] * electric_potential.pulse_width

            name = f'pw={uround(electric_potential.pulse_width, "asec", 3)}as_flu={uround(electric_potential.fluence, "Jcm2", 3)}jcm2_RB={uround(r_bound.value, bohr_radius, 3)}br_RP={r_points}_L={spec_kwargs["l_bound"]}_dt={uround(time_step, asec, 5)}as'

            spec = spec_type(name,
                             file_name = str(ii),
                             time_initial = time_initial, time_final = time_final,
                             r_points = r_points,
                             **spec_kwargs)

            specs.append(spec)

        clu.specification_check(specs)

        submit_string = clu.generate_chtc_submit_string(args.job_name, len(specs), do_checkpoints = checkpoints)
        clu.submit_check(submit_string)

        # point of no return
        shutil.rmtree(job_dir, ignore_errors = True)

        clu.create_job_subdirs(job_dir)
        clu.save_specifications(specs, job_dir)
        clu.write_specifications_info_to_file(specs, job_dir)
        clu.write_parameters_info_to_file(parameters, job_dir)

        job_info = {'name': args.job_name,
                    'job_processor_type': job_processor,  # set at top of if-name-main
                    'number_of_sims': len(specs),
                    'specification_type': specs[0].__class__,
                    'external_potential_type': specs[0].electric_potential.__class__,
                    }
        clu.write_job_info_to_file(job_info, job_dir)

        clu.write_submit_file(submit_string, job_dir)

        if not args.dry:
            clu.submit_job(job_dir)
