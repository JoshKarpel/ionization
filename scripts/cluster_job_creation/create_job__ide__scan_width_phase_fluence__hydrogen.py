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
        logger.warning('This script only supports Hydrogen bound state wavefunctions!')
        # job type options
        job_processor = iclu.IDEJobProcessor

        job_dir = os.path.join(args.dir, args.job_name)

        if os.path.exists(job_dir):
            if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
                clu.abort_job_creation()
            else:
                shutil.rmtree(job_dir)

        parameters = []

        evolution_gauge = clu.Parameter(name = 'evolution_gauge',
                                        value = clu.ask_for_input('Evolution Gauge? [LEN/VEL]', default = 'LEN'))
        if evolution_gauge == 'VEL':
            print("Haven't implemented velocity gauge yet...")
            clu.abort_job_creation()
        parameters.append(evolution_gauge)

        evolution_method = clu.Parameter(name = 'evolution_method',
                                         value = clu.ask_for_input('Evolution Method? [FE/BE/RK4/ARK4]', default = 'ARK4'))
        parameters.append(evolution_method)

        test_charge = electron_charge
        test_mass = electron_mass_reduced
        test_energy = ion.HydrogenBoundState(1, 0).energy

        if evolution_gauge.value == 'LEN':
            prefactor = ide.hydrogen_prefactor_LEN(test_charge)
        # elif evolution_gauge.value == 'VEL':
        #     pass
        else:
            raise ValueError('Unknown evolution gauge')

        parameters.append(clu.Parameter(name = 'test_charge',
                                        value = test_charge))
        parameters.append(clu.Parameter(name = 'test_mass',
                                        value = test_mass))

        parameters.append(clu.Parameter(name = 'prefactor',
                                        value = prefactor))

        parameters.append(clu.Parameter(name = 'test_energy',
                                        value = test_energy))

        if evolution_gauge.value == 'LEN':
            parameters.append(clu.Parameter(name = 'kernel',
                                            value = ide.hydrogen_kernel_LEN))
        # elif evolution_gauge.value == 'VEL':
        #     parameters.append(clu.Parameter(name = 'kernel',
        #                                     value = ide.gaussian_kernel_VEL))

        parameters.append(clu.Parameter(name = 'time_step',
                                        value = asec * clu.ask_for_input('Time Step (in as)?', default = .1, cast_to = float)))

        if evolution_method.value == 'ARK4':
            parameters.append(clu.Parameter(name = 'time_step_minimum',
                                            value = asec * clu.ask_for_input('Minimum Time Step (in as)?', default = .01, cast_to = float)))

            parameters.append(clu.Parameter(name = 'time_step_maximum',
                                            value = asec * clu.ask_for_input('Maximum Time Step (in as)?', default = 10, cast_to = float)))

            parameters.append(clu.Parameter(name = 'error_on',
                                            value = clu.ask_for_input('Fractional Truncation Error Control on a or da/dt?', default = 'da/dt', cast_to = str)))

            parameters.append(clu.Parameter(name = 'epsilon',
                                            value = clu.ask_for_input('Fractional Truncation Error Limit?', default = 1e-6, cast_to = float)))

        time_bound_in_pw = clu.Parameter(name = 'time_bound_in_pw',
                                         value = clu.ask_for_input('Time Bound (in pulse widths)?', default = 35, cast_to = float))
        parameters.append(time_bound_in_pw)

        checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
        parameters.append(clu.Parameter(name = 'checkpoints',
                                        value = checkpoints))
        if checkpoints:
            time_between_checkpoints = clu.ask_for_input('How long between checkpoints (in minutes)?', default = 60, cast_to = int)
            parameters.append(clu.Parameter(name = 'checkpoint_every',
                                            value = datetime.timedelta(minutes = time_between_checkpoints)))

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
                                          value = clu.ask_for_input('Window Time (in pulse widths)?', default = np.abs(time_bound_in_pw.value) - 5, cast_to = float))
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

        parameters.append(clu.Parameter(name = 'store_data_every',
                                        value = clu.ask_for_input('Store Data Every?', default = 1, cast_to = int)))

        print('Generating parameters...')

        spec_kwargs_list = clu.expand_parameters_to_dicts(parameters)
        specs = []

        print('Generating specifications...')

        for ii, spec_kwargs in tqdm(enumerate(spec_kwargs_list)):
            electric_potential = spec_kwargs['electric_potential']

            name = '{}_pw={}asec_flu={}Jcm2_phase={}pi'.format(
                pulse_type_q,
                uround(electric_potential.pulse_width, asec),
                uround(electric_potential.fluence, Jcm2),
                uround(electric_potential.phase, pi)
            )

            time_bound = spec_kwargs['time_bound_in_pw'] * electric_potential.pulse_width
            spec = ide.IntegroDifferentialEquationSpecification(name,
                                                                file_name = str(ii),
                                                                time_initial = -time_bound, time_final = time_bound,
                                                                **spec_kwargs)

            spec.pulse_type = pulse_type
            spec.pulse_width = electric_potential.pulse_width
            spec.fluence = electric_potential.fluence
            spec.phase = electric_potential.phase

            specs.append(spec)

        clu.specification_check(specs)

        submit_string = clu.generate_chtc_submit_string(args.job_name, len(specs), checkpoints = checkpoints)
        clu.submit_check(submit_string)

        # point of no return
        shutil.rmtree(job_dir, ignore_errors = True)

        clu.create_job_subdirs(job_dir)
        clu.save_specifications(specs, job_dir)
        clu.write_specifications_info_to_file(specs, job_dir)
        clu.write_parameters_info_to_file(parameters + pulse_parameters, job_dir)

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