import argparse
import os
import shutil
import datetime
import logging
import inspect

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

from . import core, states, potentials, ide

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InvalidChoice(Exception):
    pass


def parse_args(**kwargs):
    parser = argparse.ArgumentParser(**kwargs)
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

    return parser.parse_args()


def get_log_manager(args, **kwargs):
    return si.utils.LogManager('simulacra', 'ionization', 'jobutils', stdout_level = 31 - ((args.verbosity + 1) * 10), **kwargs)


def get_job_dir(args):
    return os.path.join(args.dir, args.job_name)


def check_job_dir(args):
    job_dir_path = get_job_dir(args)

    if os.path.exists(job_dir_path):
        if not args.overwrite and not clu.ask_for_bool('A job with that name already exists. Overwrite?', default = 'No'):
            clu.abort_job_creation()


def ask_mesh_type():
    mesh_kwargs = {}

    mesh_type = clu.ask_for_input('Mesh Type (cyl | sph | harm)', default = 'harm', cast_to = str)

    try:
        if mesh_type == 'cyl':
            spec_type = core.CylindricalSliceSpecification

            mesh_kwargs['z_bound'] = u.bohr_radius * clu.ask_for_input('Z Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['rho_bound'] = u.bohr_radius * clu.ask_for_input('Rho Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['z_points'] = 2 * (mesh_kwargs['z_bound'] / u.bohr_radius) * clu.ask_for_input('Z Points per Bohr Radii', default = 20, cast_to = int)
            mesh_kwargs['rho_points'] = (mesh_kwargs['rho_bound'] / u.bohr_radius) * clu.ask_for_input('Rho Points per Bohr Radii', default = 20, cast_to = int)

            mesh_kwargs['outer_radius'] = max(mesh_kwargs['z_bound'], mesh_kwargs['rho_bound'])

            memory_estimate = (128 / 8) * mesh_kwargs['z_points'] * mesh_kwargs['rho_points']

        elif mesh_type == 'sph':
            spec_type = core.SphericalSliceSpecification

            mesh_kwargs['r_bound'] = u.bohr_radius * clu.ask_for_input('R Bound (Bohr radii)', default = 30, cast_to = float)
            mesh_kwargs['r_points'] = (mesh_kwargs['r_bound'] / u.bohr_radius) * clu.ask_for_input('R Points per Bohr Radii', default = 40, cast_to = int)
            mesh_kwargs['theta_points'] = clu.ask_for_input('Theta Points', default = 100, cast_to = int)

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['theta_points']

        elif mesh_type == 'harm':
            spec_type = core.SphericalHarmonicSpecification

            r_bound = clu.ask_for_input('R Bound (Bohr radii)', default = 200, cast_to = float)
            mesh_kwargs['r_points'] = r_bound * clu.ask_for_input('R Points per Bohr Radii', default = 10, cast_to = int)
            mesh_kwargs['l_bound'] = clu.ask_for_input('l points', default = 500, cast_to = int)

            mesh_kwargs['r_bound'] = u.bohr_radius * r_bound

            mesh_kwargs['outer_radius'] = mesh_kwargs['r_bound']

            mesh_kwargs['snapshot_type'] = core.SphericalHarmonicSnapshot

            memory_estimate = (128 / 8) * mesh_kwargs['r_points'] * mesh_kwargs['l_bound']

        else:
            raise ValueError('Mesh type {} not found!'.format(mesh_type))

        logger.warning('Predicted memory usage per Simulation is >{}'.format(si.utils.bytes_to_str(memory_estimate)))

        return spec_type, mesh_kwargs
    except ValueError:
        ask_mesh_type()


def ask_mask__radial_cosine(parameters, mesh_kwargs):
    outer_radius_default = mesh_kwargs['outer_radius'] / u.bohr_radius

    inner = u.bohr_radius * clu.ask_for_input(
        'Mask Inner Radius (in Bohr radii)?',
        default = np.ceil(outer_radius_default * .8),
        cast_to = float
    )
    outer = u.bohr_radius * clu.ask_for_input(
        'Mask Outer Radius (in Bohr radii)?',
        default = np.ceil(outer_radius_default),
        cast_to = float
    )
    smoothness = clu.ask_for_input('Mask Smoothness?', default = 8, cast_to = int)

    mask = clu.Parameter(
        name = 'mask',
        value = potentials.RadialCosineMask(
            inner_radius = inner,
            outer_radius = outer,
            smoothness = smoothness,
        ))
    parameters.append(mask)


def ask_initial_state_for_hydrogen_sim(parameters):
    initial_state = clu.Parameter(
        name = 'initial_state',
        value = states.HydrogenBoundState(
            n = clu.ask_for_input('Initial State n?', default = 1, cast_to = int),
            l = clu.ask_for_input('Initial State l?', default = 0, cast_to = int),
        ))
    parameters.append(initial_state)

    return initial_state


def ask_numeric_eigenstate_basis(parameters, *, spec_type):
    numeric_basis_q = clu.ask_for_bool('Use numeric eigenstate basis?', default = True)
    if numeric_basis_q:
        parameters.append(
            clu.Parameter(
                name = 'use_numeric_eigenstates',
                value = True,
            ))

        max_energy = u.eV * clu.ask_for_input('Numeric Eigenstate Max Energy (in eV)?', default = 20, cast_to = float)
        parameters.append(
            clu.Parameter(
                name = 'numeric_eigenstate_max_energy',
                value = max_energy,
            ))

        if spec_type == core.SphericalHarmonicSpecification:
            max_angular_momentum = clu.ask_for_input('Numeric Eigenstate Maximum l?', default = 20, cast_to = int)
            parameters.append(
                clu.Parameter(
                    name = 'numeric_eigenstate_max_angular_momentum',
                    value = max_angular_momentum,
                ))

            return max_energy, max_angular_momentum
        else:
            return max_energy, None


def ask_time_step(parameters):
    parameters.append(
        clu.Parameter(
            name = 'time_step',
            value = u.asec * clu.ask_for_input('Time Step (in as)?', default = 1, cast_to = float),
        ))


def ask_time_evolution_by_pulse_widths():
    time_initial_in_pw = clu.ask_for_input('Initial Time (in pulse widths)?', default = -35, cast_to = float)
    time_final_in_pw = clu.ask_for_input('Final Time (in pulse widths)?', default = 35, cast_to = float)
    extra_time = u.asec * clu.ask_for_input('Extra Time (in as)?', default = 0, cast_to = float)

    return time_initial_in_pw, time_final_in_pw, extra_time


def ask_evolution_gauge(parameters, *, spec_type):
    choices = sorted(list(spec_type.evolution_gauge.choices))
    gauge = clu.ask_for_input(f'Evolution Gauge? [{"/".join(choices)}]', default = choices[0])
    if gauge not in choices:
        raise InvalidChoice(f'{gauge} is not one of {choices}')
    parameters.append(
        clu.Parameter(
            name = 'evolution_gauge',
            value = gauge,
        ))

    return gauge


def ask_evolution_method_ide(parameters, *, spec_type):
    choices = {
        'FE': ide.ForwardEulerMethod,
        'BE': ide.BackwardEulerMethod,
        'TRAP': ide.TrapezoidMethod,
        'RK4': ide.RungeKuttaFourMethod,
    }
    method_key = clu.ask_for_input(f'Evolution Method? [{"/".join(choices.keys())}]', default = 'RK4')
    try:
        method = choices[method_key]()
    except KeyError:
        raise InvalidChoice(f'{method} is not one of {choices}')

    parameters.append(
        clu.Parameter(
            name = 'evolution_method',
            value = method,
        ))

    return method


def ask_evolution_method_tdse(parameters, *, spec_type):
    choices = sorted(list(spec_type.evolution_method.choices))
    method = clu.ask_for_input(f'Evolution Method? [{"/".join(choices)}]', default = 'SO' if 'SO' in choices else choices[0])
    if method not in choices:
        raise InvalidChoice(f'{method} is not one of {choices}')

    parameters.append(
        clu.Parameter(
            name = 'evolution_method',
            value = method,
        ))

    return method


def ask_ide_kernel(parameters):
    choices = {
        'hydrogen': ide.LengthGaugeHydrogenKernel,
        'hydrogen_with_cc': ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction,
    }
    kernel_key = clu.ask_for_input(f'IDE Kernel? [{"/".join(choices)}]', default = 'hydrogen')
    try:
        kernel = choices[kernel_key]()
    except KeyError:
        raise InvalidChoice(f'{kernel_key} is not one of {choices.keys()}')

    parameters.append(
        clu.Parameter(
            name = 'kernel',
            value = kernel,
        ))

    return kernel


PULSE_NAMES_TO_TYPES = {
    'sinc': potentials.SincPulse,
    'gaussian': potentials.GaussianPulse,
    'sech': potentials.SechPulse,
    'cos2': potentials.CosSquaredPulse,
}

PULSE_TYPE_TO_WINDOW_TIME_CORRECTIONS = {
    potentials.SincPulse: 5,
    potentials.GaussianPulse: 1,
    potentials.SechPulse: 1,
    potentials.CosSquaredPulse: 0,
}


def ask_pulse_widths(pulse_parameters):
    pulse_width = clu.Parameter(
        name = 'pulse_width',
        value = u.asec * np.array(clu.ask_for_eval('Pulse Widths (in as)?', default = '[50, 100, 200, 400, 800]')),
        expandable = True
    )
    pulse_parameters.append(pulse_width)


def ask_pulse_fluences(pulse_parameters):
    fluence = clu.Parameter(
        name = 'fluence',
        value = u.Jcm2 * np.array(clu.ask_for_eval('Pulse Fluence (in J/cm^2)?', default = '[.01, .1, 1, 10, 20]')),
        expandable = True
    )
    pulse_parameters.append(fluence)


def ask_pulse_phases(pulse_parameters):
    phases = clu.Parameter(
        name = 'phase',
        value = np.array(clu.ask_for_eval('Pulse CEP (in rad)?', default = '[0, pi / 4, pi / 2]')),
        expandable = True
    )
    pulse_parameters.append(phases)


def ask_pulse_omega_mins(pulse_parameters):
    omega_mins = clu.Parameter(
        name = 'omega_min',
        value = u.twopi * u.THz * np.array(clu.ask_for_eval('Pulse Frequency Minimum? (in THz)', default = '[30]')),
        expandable = True)
    pulse_parameters.append(omega_mins)


def ask_pulse_omega_carriers(pulse_parameters):
    raise NotImplementedError


def ask_pulse_keldysh_parameters(pulse_parameters):
    raise NotImplementedError


def ask_pulse_amplitudes(pulse_parameters):
    amplitude_prefactors = clu.Parameter(
        name = 'amplitude',
        value = u.atomic_electric_field * np.array(clu.ask_for_eval('Pulse Amplitudes? (in AEF)', default = '[.01, .05, .1, .5, 1, 2]')),
        expandable = True)
    pulse_parameters.append(amplitude_prefactors)


def ask_pulse_power_exclusion(pulse_parameters):
    raise NotImplementedError


def ask_pulse_number_of_pulse_widths(pulse_parameters):
    number_of_pulse_widths = clu.Parameter(
        name = 'number_of_cycles',
        value = np.array(clu.ask_for_eval('Number of Pulse Widths to count Cycles over?', default = '[3]')),
        expandable = True)
    pulse_parameters.append(number_of_pulse_widths)


def ask_pulse_number_of_cycles(pulse_parameters):
    number_of_cycles = clu.Parameter(
        name = 'number_of_cycles',
        value = np.array(clu.ask_for_eval('Number of Cycles?', default = '[2, 3, 4]')),
        expandable = True)
    pulse_parameters.append(number_of_cycles)


CONSTRUCTOR_ARG_TO_ASK = {
    'pulse_width': ask_pulse_widths,
    'fluence': ask_pulse_fluences,
    'phase': ask_pulse_phases,
    'omega_min': ask_pulse_omega_mins,
    'omega_carriers': ask_pulse_omega_carriers,
    'keldysh_parameter': ask_pulse_keldysh_parameters,
    'amplitude': ask_pulse_amplitudes,
    'number_of_pulse_widths': ask_pulse_number_of_pulse_widths,
    'number_of_cycles': ask_pulse_number_of_cycles,
}


def ask_pulse_window(*, pulse_type, time_initial_in_pw, time_final_in_pw):
    window_time_guess = min(abs(time_initial_in_pw), abs(time_final_in_pw)) - PULSE_TYPE_TO_WINDOW_TIME_CORRECTIONS[pulse_type]

    window_time_in_pw = clu.ask_for_input('Window Time (in pulse widths)?', default = window_time_guess, cast_to = float)
    window_width_in_pw = clu.ask_for_input('Window Width (in pulse widths)?', default = 0.2, cast_to = float)

    return window_time_in_pw, window_width_in_pw


def construct_pulses(parameters, *, time_initial_in_pw, time_final_in_pw):
    pulse_parameters = []

    pulse_type = PULSE_NAMES_TO_TYPES[clu.ask_for_input('Pulse Type? (sinc | gaussian | sech | cos2)', default = 'sinc')]
    constructor_names = (name.replace('from_', '') for name in pulse_type.__dict__ if 'from_' in name)
    constructor_name = clu.ask_for_input(f'Pulse Constructor? ({" | ".join(constructor_names)})', default = 'omega_min')
    constructor = getattr(pulse_type, f'from_{constructor_name}')

    constructor_argspec = inspect.getfullargspec(constructor)
    if constructor_argspec.varargs is not None:  # alias for default constructor, super implicit....
        constructor_args = inspect.getfullargspec(pulse_type.__init__).args
    else:
        constructor_args = constructor_argspec.args

    asks = (CONSTRUCTOR_ARG_TO_ASK[arg] for arg in CONSTRUCTOR_ARG_TO_ASK if arg in constructor_args)
    for ask in asks:
        ask(pulse_parameters)

    window_time_in_pw, window_width_in_pw = ask_pulse_window(
        pulse_type = pulse_type,
        time_initial_in_pw = time_initial_in_pw,
        time_final_in_pw = time_final_in_pw
    )

    print('Generating pulses...')
    pulses = tuple(
        constructor(
            **d,
            window = potentials.SymmetricExponentialTimeWindow(
                window_time = d['pulse_width'] * window_time_in_pw,
                window_width = d['pulse_width'] * window_width_in_pw,
            ),
        )
        for d in tqdm(clu.expand_parameters_to_dicts(pulse_parameters), ascii = True)
    )
    parameters.append(
        clu.Parameter(
            name = 'electric_potential',
            value = pulses,
            expandable = True
        )
    )

    parameters.append(
        clu.Parameter(
            name = 'electric_potential_dc_correction',
            value = clu.ask_for_bool('Perform Electric Field DC Correction?', default = True)
        ))

    return pulse_parameters


def ask_checkpoints(parameters):
    do_checkpoints = clu.ask_for_bool('Checkpoints?', default = True)
    parameters.append(
        clu.Parameter(
            name = 'checkpoints',
            value = do_checkpoints
        ))

    if do_checkpoints:
        time_between_checkpoints = clu.ask_for_input('How long between checkpoints (in minutes)?', default = 60, cast_to = int)
        parameters.append(
            clu.Parameter(
                name = 'checkpoint_every',
                value = datetime.timedelta(minutes = time_between_checkpoints)
            ))

    return do_checkpoints


def ask_data_storage_tdse(parameters, *, spec_type):
    parameters.append(
        clu.Parameter(
            name = 'store_data_every',
            value = clu.ask_for_input('Store Data Every n Time Steps', default = -1, cast_to = int),
        ))

    names_questions_defaults = [
        ('store_radial_position_expectation_value', 'Store Radial Position EV vs. Time?', True),
        ('store_electric_dipole_moment_expectation_value', 'Store Electric Dipole Moment EV vs. Time?', True),
        ('store_energy_expectation_value', 'Store Energy EV vs. Time?', True),
        ('store_norm_diff_mask', 'Store Difference in Norm caused by Mask vs. Time?', False),
    ]
    if spec_type == core.SphericalHarmonicSpecification:
        names_questions_defaults += [
            ('store_radial_probability_current', 'Store Radial Probability Current vs. Time?', False),
            ('store_norm_by_l', 'Store Norm-by-L?', False),
        ]

    for name, question, default in names_questions_defaults:
        parameters.append(
            clu.Parameter(
                name = name,
                value = clu.ask_for_bool(question, default = default)
            ))


def ask_data_storage_ide(parameters, *, spec_type):
    parameters.append(
        clu.Parameter(
            name = 'store_data_every',
            value = clu.ask_for_input('Store Data Every n Time Steps', default = -1, cast_to = int),
        ))


POTENTIAL_ATTRS = [
    'pulse_width',
    'phase',
    'fluence',
    'amplitude',
    'number_of_cycles',
    'omega_carrier',
]


def transfer_potential_attrs_to_spec(electric_potential, spec):
    spec.pulse_type = electric_potential.__class__.__name__
    for attr in POTENTIAL_ATTRS:
        try:
            setattr(spec, attr, getattr(electric_potential, attr))
        except AttributeError:
            pass


def create_job_files(*, args, specs, do_checkpoints, parameters, pulse_parameters, job_processor_type):
    job_dir = get_job_dir(args)

    clu.specification_check(specs)

    submit_string = clu.generate_chtc_submit_string(
        args.job_name,
        len(specs),
        do_checkpoints = do_checkpoints
    )
    clu.submit_check(submit_string)

    # point of no return
    shutil.rmtree(job_dir, ignore_errors = True)

    clu.create_job_subdirs(job_dir)
    clu.save_specifications(specs, job_dir)
    clu.write_specifications_info_to_file(specs, job_dir)
    clu.write_parameters_info_to_file(parameters + pulse_parameters, job_dir)

    job_info = {
        'name': args.job_name,
        'job_processor_type': job_processor_type,  # set at top of if-name-main
        'number_of_sims': len(specs),
        'specification_type': specs[0].__class__,
        'external_potential_type': specs[0].electric_potential.__class__,
    }
    clu.write_job_info_to_file(job_info, job_dir)

    clu.write_submit_file(submit_string, job_dir)
