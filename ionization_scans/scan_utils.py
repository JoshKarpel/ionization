from pathlib import Path
import itertools
import inspect
import datetime

import numpy as np
from tqdm import tqdm

import simulacra as si
import simulacra.units as u

import ionization as ion

import htmap

# SHARED QUESTIONS


def ask_for_tag():
    tag = si.ask_for_input("Map Tag?", default=None)
    if tag is None:
        raise ValueError("tag cannot be None")
    return tag


def ask_mesh_type():
    mesh_kwargs = {}

    mesh_type = si.ask_for_input(
        "Mesh Type [cyl | sph | harm]", default="harm", callback=str
    )

    if mesh_type == "cyl":
        spec_type = ion.mesh.CylindricalSliceSpecification

        mesh_kwargs["z_bound"] = u.bohr_radius * si.ask_for_input(
            "Z Bound (Bohr radii)", default=30, callback=float
        )
        mesh_kwargs["rho_bound"] = u.bohr_radius * si.ask_for_input(
            "Rho Bound (Bohr radii)", default=30, callback=float
        )
        mesh_kwargs["z_points"] = (
            2
            * (mesh_kwargs["z_bound"] / u.bohr_radius)
            * si.ask_for_input("Z Points per Bohr Radii", default=20, callback=int)
        )
        mesh_kwargs["rho_points"] = (
            mesh_kwargs["rho_bound"] / u.bohr_radius
        ) * si.ask_for_input("Rho Points per Bohr Radii", default=20, callback=int)

        mesh_kwargs["outer_radius"] = max(
            mesh_kwargs["z_bound"], mesh_kwargs["rho_bound"]
        )

        memory_estimate = (
            (128 / 8) * mesh_kwargs["z_points"] * mesh_kwargs["rho_points"]
        )

    elif mesh_type == "sph":
        spec_type = ion.mesh.SphericalSliceSpecification

        mesh_kwargs["r_bound"] = u.bohr_radius * si.ask_for_input(
            "R Bound (Bohr radii)", default=30, callback=float
        )
        mesh_kwargs["r_points"] = (
            mesh_kwargs["r_bound"] / u.bohr_radius
        ) * si.ask_for_input("R Points per Bohr Radii", default=40, callback=int)
        mesh_kwargs["theta_points"] = si.ask_for_input(
            "Theta Points", default=100, callback=int
        )

        mesh_kwargs["outer_radius"] = mesh_kwargs["r_bound"]

        memory_estimate = (
            (128 / 8) * mesh_kwargs["r_points"] * mesh_kwargs["theta_points"]
        )

    elif mesh_type == "harm":
        spec_type = ion.mesh.SphericalHarmonicSpecification

        r_bound = si.ask_for_input("R Bound (Bohr radii)", default=200, callback=float)
        mesh_kwargs["r_points"] = r_bound * si.ask_for_input(
            "R Points per Bohr Radii", default=10, callback=int
        )
        mesh_kwargs["l_bound"] = si.ask_for_input("l points", default=500, callback=int)

        mesh_kwargs["r_bound"] = u.bohr_radius * r_bound

        mesh_kwargs["outer_radius"] = mesh_kwargs["r_bound"]

        mesh_kwargs["snapshot_type"] = ion.mesh.SphericalHarmonicSnapshot

        memory_estimate = (128 / 8) * mesh_kwargs["r_points"] * mesh_kwargs["l_bound"]

    else:
        print(f"Mesh type {mesh_type} not found!")
        return ask_mesh_type()

    print(
        f"Predicted memory usage per Simulation is at least {si.utils.bytes_to_str(memory_estimate)}"
    )

    return spec_type, mesh_kwargs


def ask_mask__radial_cosine(parameters, mesh_kwargs):
    outer_radius_default = mesh_kwargs["outer_radius"] / u.bohr_radius

    inner = u.bohr_radius * si.ask_for_input(
        "Mask Inner Radius (in Bohr radii)?",
        default=np.ceil(outer_radius_default * 0.8),
        callback=float,
    )
    outer = u.bohr_radius * si.ask_for_input(
        "Mask Outer Radius (in Bohr radii)?",
        default=np.ceil(outer_radius_default),
        callback=float,
    )
    smoothness = si.ask_for_input("Mask Smoothness?", default=8, callback=int)

    mask = si.Parameter(
        name="mask",
        value=ion.potentials.RadialCosineMask(
            inner_radius=inner, outer_radius=outer, smoothness=smoothness
        ),
    )
    parameters.append(mask)


def ask_initial_state_for_hydrogen_sim(parameters):
    initial_state = si.Parameter(
        name="initial_state",
        value=ion.states.HydrogenBoundState(
            n=si.ask_for_input("Initial State n?", default=1, callback=int),
            l=si.ask_for_input("Initial State l?", default=0, callback=int),
        ),
    )
    parameters.append(initial_state)

    return initial_state


def ask_numeric_eigenstate_basis(parameters, *, spec_type):
    numeric_basis_q = si.ask_for_bool("Use numeric eigenstate basis?", default=True)
    if numeric_basis_q:
        parameters.append(si.Parameter(name="use_numeric_eigenstates", value=True))

        max_energy = u.eV * si.ask_for_input(
            "Numeric Eigenstate Max Energy (in eV)?", default=20, callback=float
        )
        parameters.append(
            si.Parameter(name="numeric_eigenstate_max_energy", value=max_energy)
        )

        if spec_type == ion.mesh.SphericalHarmonicSpecification:
            max_angular_momentum = si.ask_for_input(
                "Numeric Eigenstate Maximum l?", default=20, callback=int
            )
            parameters.append(
                si.Parameter(
                    name="numeric_eigenstate_max_angular_momentum",
                    value=max_angular_momentum,
                )
            )

            return max_energy, max_angular_momentum
        else:
            return max_energy, None


def ask_time_step(parameters):
    parameters.append(
        si.Parameter(
            name="time_step",
            value=u.asec
            * si.ask_for_input("Time Step (in as)?", default=1, callback=float),
        )
    )


def ask_time_evolution():
    time_initial_in_pw = si.ask_for_input(
        "Initial Time (in pulse widths)?", default=-35, callback=float
    )
    time_final_in_pw = si.ask_for_input(
        "Final Time (in pulse widths)?", default=35, callback=float
    )
    extra_time = u.asec * si.ask_for_input(
        "Extra Time (in as)?", default=0, callback=float
    )

    return time_initial_in_pw, time_final_in_pw, extra_time


def ask_mesh_operators(parameters, *, spec_type):
    choices_by_spec_type = {
        ion.mesh.LineSpecification: {
            ion.Gauge.LENGTH.value: ion.mesh.LineLengthGaugeOperators,
            ion.Gauge.VELOCITY.value: ion.mesh.LineVelocityGaugeOperators,
        },
        ion.mesh.CylindricalSliceSpecification: {
            ion.Gauge.LENGTH.value: ion.mesh.CylindricalSliceLengthGaugeOperators
        },
        ion.mesh.SphericalSliceSpecification: {
            ion.Gauge.LENGTH.value: ion.mesh.SphericalSliceLengthGaugeOperators
        },
        ion.mesh.SphericalHarmonicSpecification: {
            ion.Gauge.LENGTH.value: ion.mesh.SphericalHarmonicLengthGaugeOperators,
            ion.Gauge.VELOCITY.value: ion.mesh.SphericalHarmonicVelocityGaugeOperators,
        },
    }
    choices = choices_by_spec_type[spec_type]
    choice = si.ask_for_input(
        f'Mesh Operators? [{" | ".join(choices.keys())}]',
        default=ion.Gauge.LENGTH.value,
    )
    try:
        method = choices[choice]()
    except KeyError:
        raise ion.exceptions.InvalidChoice(f"{choice} is not one of {choices}")

    parameters.append(si.Parameter(name="evolution_method", value=method))

    return method


def ask_evolution_method_ide(parameters):
    choices = {
        "FE": ion.ide.ForwardEulerMethod,
        "BE": ion.ide.BackwardEulerMethod,
        "TRAP": ion.ide.TrapezoidMethod,
        "RK4": ion.ide.RungeKuttaFourMethod,
    }
    key = si.ask_for_input(
        f'Evolution Method? [{" | ".join(choices.keys())}]', default="RK4"
    )
    try:
        method = choices[key]()
    except KeyError:
        raise ion.exceptions.InvalidChoice(f"{method} is not one of {choices}")

    parameters.append(si.Parameter(name="evolution_method", value=method))

    return method


def ask_evolution_method_tdse(parameters):
    choices = {
        "ADI": ion.mesh.AlternatingDirectionImplicit,
        "SO": ion.mesh.SplitInteractionOperator,
    }
    key = si.ask_for_input(f'Evolution Method? [{" | ".join(choices)}]', default="SO")
    try:
        method = choices[key]()
    except KeyError:
        raise ion.exceptions.InvalidChoice(f"{method} is not one of {choices}")

    parameters.append(si.Parameter(name="evolution_method", value=method))

    return method


def ask_ide_kernel(parameters):
    choices = {
        "hydrogen": ion.ide.LengthGaugeHydrogenKernel,
        "hydrogen_with_cc": ion.ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction,
    }
    kernel_key = si.ask_for_input(
        f'IDE Kernel? [{" | ".join(choices)}]', default="hydrogen"
    )
    try:
        kernel = choices[kernel_key]()
    except KeyError:
        raise ion.exceptions.InvalidChoice(
            f"{kernel_key} is not one of {choices.keys()}"
        )

    parameters.append(si.Parameter(name="kernel", value=kernel))

    return kernel


def ask_ide_tunneling(parameters):
    choices = {
        cls.__name__.replace("Rate", ""): cls
        for cls in ion.tunneling.TUNNELING_MODEL_TYPES
    }
    model_key = si.ask_for_input(
        f'Tunneling Model? [{" | ".join(choices)}]', default=tuple(choices.keys())[0]
    )

    try:
        cls = choices[model_key]
    except KeyError:
        raise ion.exceptions.InvalidChoice(
            f"{model_key} is not one of {choices.keys()}"
        )

    argspec = inspect.getfullargspec(cls.__init__)
    arg_names = argspec.args[1:]
    arg_defaults = argspec.defaults
    if len(arg_names) > 0:
        args = {
            name: si.ask_for_eval(f"Value for {name}?", default=repr(default))
            for name, default in reversed(
                tuple(
                    itertools.zip_longest(reversed(arg_names), reversed(arg_defaults))
                )
            )
        }
        model = cls(**args)
    else:
        model = cls()

    parameters.append(si.Parameter(name="tunneling_model", value=model))

    return model


PULSE_NAMES_TO_TYPES = {
    "sinc": ion.potentials.SincPulse,
    "gaussian": ion.potentials.GaussianPulse,
    "sech": ion.potentials.SechPulse,
    "cos2": ion.potentials.CosSquaredPulse,
}

PULSE_TYPE_TO_WINDOW_TIME_CORRECTIONS = {
    ion.potentials.SincPulse: 5,
    ion.potentials.GaussianPulse: 1,
    ion.potentials.SechPulse: 1,
    ion.potentials.CosSquaredPulse: 0,
}


def ask_pulse_widths(pulse_parameters):
    pulse_width = si.Parameter(
        name="pulse_width",
        value=u.asec
        * np.array(
            si.ask_for_eval("Pulse Widths (in as)?", default="[50, 100, 200, 400, 800]")
        ),
        expandable=True,
    )
    pulse_parameters.append(pulse_width)


def ask_pulse_fluences(pulse_parameters):
    fluence = si.Parameter(
        name="fluence",
        value=u.Jcm2
        * np.array(
            si.ask_for_eval(
                "Pulse Fluence (in J/cm^2)?", default="[.01, .1, 1, 10, 20]"
            )
        ),
        expandable=True,
    )
    pulse_parameters.append(fluence)


def ask_pulse_phases(pulse_parameters):
    phases = si.Parameter(
        name="phase",
        value=np.array(
            si.ask_for_eval("Pulse CEP (in rad)?", default="[0, u.pi / 4, u.pi / 2]")
        ),
        expandable=True,
    )
    pulse_parameters.append(phases)


def ask_pulse_omega_mins(pulse_parameters):
    omega_mins = si.Parameter(
        name="omega_min",
        value=u.twopi
        * u.THz
        * np.array(
            si.ask_for_eval("Pulse Frequency Minimum? (in THz)", default="[30]")
        ),
        expandable=True,
    )
    pulse_parameters.append(omega_mins)


def ask_pulse_omega_carriers(pulse_parameters):
    raise NotImplementedError


def ask_pulse_keldysh_parameters(pulse_parameters):
    raise NotImplementedError


def ask_pulse_amplitudes(pulse_parameters):
    amplitude_prefactors = si.Parameter(
        name="amplitude",
        value=u.atomic_electric_field
        * np.array(
            si.ask_for_eval(
                "Pulse Amplitudes? (in AEF)", default="[.01, .05, .1, .5, 1, 2]"
            )
        ),
        expandable=True,
    )
    pulse_parameters.append(amplitude_prefactors)


def ask_pulse_power_exclusion(pulse_parameters):
    raise NotImplementedError


def ask_pulse_number_of_pulse_widths(pulse_parameters):
    number_of_pulse_widths = si.Parameter(
        name="number_of_pulse_widths",
        value=np.array(
            si.ask_for_eval(
                "Number of Pulse Widths to count Cycles over?", default="[3]"
            )
        ),
        expandable=True,
    )
    pulse_parameters.append(number_of_pulse_widths)


def ask_pulse_number_of_cycles(pulse_parameters):
    number_of_cycles = si.Parameter(
        name="number_of_cycles",
        value=np.array(si.ask_for_eval("Number of Cycles?", default="[2, 3, 4]")),
        expandable=True,
    )
    pulse_parameters.append(number_of_cycles)


CONSTRUCTOR_ARG_TO_ASK = {
    "pulse_width": ask_pulse_widths,
    "fluence": ask_pulse_fluences,
    "phase": ask_pulse_phases,
    "omega_min": ask_pulse_omega_mins,
    "omega_carriers": ask_pulse_omega_carriers,
    "keldysh_parameter": ask_pulse_keldysh_parameters,
    "amplitude": ask_pulse_amplitudes,
    "number_of_pulse_widths": ask_pulse_number_of_pulse_widths,
    "number_of_cycles": ask_pulse_number_of_cycles,
}


def ask_pulse_window(*, pulse_type, time_initial_in_pw, time_final_in_pw):
    window_time_guess = (
        min(abs(time_initial_in_pw), abs(time_final_in_pw))
        - PULSE_TYPE_TO_WINDOW_TIME_CORRECTIONS[pulse_type]
    )

    window_time_in_pw = si.ask_for_input(
        "Window Time (in pulse widths)?", default=window_time_guess, callback=float
    )
    window_width_in_pw = si.ask_for_input(
        "Window Width (in pulse widths)?", default=0.2, callback=float
    )

    return window_time_in_pw, window_width_in_pw


def construct_pulses(parameters, *, time_initial_in_pw, time_final_in_pw):
    pulse_parameters = []

    pulse_type = PULSE_NAMES_TO_TYPES[
        si.ask_for_input("Pulse Type? [sinc | gaussian | sech | cos2]", default="sinc")
    ]
    constructor_names = (
        name.replace("from_", "")
        for name in pulse_type.__dict__
        if name.startswith("from_")
    )
    constructor_name = si.ask_for_input(
        f'Pulse Constructor? [{" | ".join(constructor_names)}]', default="omega_min"
    )
    constructor = getattr(pulse_type, f"from_{constructor_name}")

    constructor_argspec = inspect.getfullargspec(constructor)
    if (
        constructor_argspec.varargs is not None
    ):  # alias for default constructor, super implicit....
        constructor_args = inspect.getfullargspec(pulse_type.__init__).args
    else:
        constructor_args = constructor_argspec.args

    asks = (
        CONSTRUCTOR_ARG_TO_ASK[arg]
        for arg in CONSTRUCTOR_ARG_TO_ASK
        if arg in constructor_args
    )
    for ask in asks:
        ask(pulse_parameters)

    window_time_in_pw, window_width_in_pw = ask_pulse_window(
        pulse_type=pulse_type,
        time_initial_in_pw=time_initial_in_pw,
        time_final_in_pw=time_final_in_pw,
    )

    print("Generating pulses...")
    pulses = tuple(
        constructor(
            **d,
            window=ion.potentials.LogisticWindow(
                window_time=d["pulse_width"] * window_time_in_pw,
                window_width=d["pulse_width"] * window_width_in_pw,
            ),
        )
        for d in tqdm(si.expand_parameters(pulse_parameters), ascii=True)
    )
    parameters.append(
        si.Parameter(name="electric_potential", value=pulses, expandable=True)
    )

    parameters.append(
        si.Parameter(
            name="electric_potential_dc_correction",
            value=si.ask_for_bool(
                "Perform Electric Field DC Correction?", default=True
            ),
        )
    )

    parameters.append(
        si.Parameter(
            name="electric_potential_fluence_correction",
            value=si.ask_for_bool(
                "Perform Electric Field Fluence Correction?", default=False
            ),
        )
    )

    return pulse_parameters


def ask_checkpoints(parameters):
    do_checkpoints = si.ask_for_bool("Checkpoints?", default=True)
    parameters.append(si.Parameter(name="checkpoints", value=do_checkpoints))

    if do_checkpoints:
        time_between_checkpoints = si.ask_for_input(
            "How long between checkpoints (in minutes)?", default=60, callback=int
        )
        parameters.append(
            si.Parameter(
                name="checkpoint_every",
                value=datetime.timedelta(minutes=time_between_checkpoints),
            )
        )

    return do_checkpoints


def ask_data_storage_tdse(parameters, *, spec_type):
    parameters.append(
        si.Parameter(
            name="store_data_every",
            value=si.ask_for_input(
                "Store Data Every n Time Steps", default=-1, callback=int
            ),
        )
    )

    datastores_questions_defaults = [
        (ion.mesh.Fields, "Store Electric Field and Vector Potential vs. Time?", True),
        (ion.mesh.Norm, "Store Wavefunction Norm vs. Time?", True),
        (ion.mesh.InnerProducts, "Store Wavefunction Inner Products vs. Time?", True),
        (
            ion.mesh.InternalEnergyExpectationValue,
            "Store Internal Energy Expectation Value vs. Time?",
            False,
        ),
        (
            ion.mesh.TotalEnergyExpectationValue,
            "Store Total Energy Expectation Value vs. Time?",
            False,
        ),
        (ion.mesh.ZExpectationValue, "Store Z Expectation Value vs. Time?", False),
        (ion.mesh.RExpectationValue, "Store R Expectation Value vs. Time?", False),
        (ion.mesh.NormWithinRadius, "Store Norm Within Radius vs. Time?", False),
    ]
    if spec_type == ion.mesh.SphericalHarmonicSpecification:
        datastores_questions_defaults += [
            (
                ion.mesh.DirectionalRadialProbabilityCurrent,
                "Store Radial Probability Current vs. Time?",
                False,
            ),
            (ion.mesh.NormBySphericalHarmonic, "Store Norm-by-L?", False),
        ]

    datastores = []
    for cls, question, default in datastores_questions_defaults:
        if not si.ask_for_bool(question, default=default):
            continue
        argspec = inspect.getfullargspec(cls.__init__)
        arg_names = argspec.args[1:]
        arg_defaults = argspec.defaults
        if len(arg_names) > 0:
            args = {
                name: si.ask_for_eval(f"Value for {name}?", default=default)
                for name, default in reversed(
                    tuple(
                        itertools.zip_longest(
                            reversed(arg_names), reversed(arg_defaults)
                        )
                    )
                )
            }
            datastore = cls(**args)
        else:
            datastore = cls()
        datastores.append(datastore)

    parameters.append(si.Parameter(name="datastores", value=datastores))


def ask_data_storage_ide(parameters, *, spec_type):
    parameters.append(
        si.Parameter(
            name="store_data_every",
            value=si.ask_for_input(
                "Store Data Every n Time Steps", default=-1, callback=int
            ),
        )
    )


def transfer_potential_attrs_to_spec(electric_potential, spec):
    spec.pulse_type = electric_potential.__class__.__name__
    for attr in ion.analysis.POTENTIAL_ATTRS:
        try:
            setattr(spec, attr, getattr(electric_potential, attr))
        except AttributeError:
            pass


@htmap.mapped(map_options=htmap.MapOptions(custom_options={"is_resumable": "true"}))
def run(spec):
    sim_path = Path.cwd() / f"{spec.name}.sim"

    try:
        sim = si.Simulation.load(str(sim_path))
        print(f"Recovered checkpoint from {sim_path}")
        print(
            f"Checkpoint size is {si.utils.bytes_to_str(si.utils.get_file_size(sim_path))}"
        )
    except (FileNotFoundError, EOFError):
        sim = spec.to_sim()
        print("No checkpoint found")

    print(sim.info())

    sim.run(checkpoint_callback=htmap.checkpoint)

    print(sim.info())

    if isinstance(sim, ion.mesh.MeshSimulation):
        for state in sim.spec.test_states:
            state.g = None
        sim.mesh = None

    return sim


def ask_htmap_settings():
    docker_image = si.ask_for_input("Docker image (repository:tag)?")
    htmap.settings["DOCKER.IMAGE"] = docker_image
    htmap.settings["SINGULARITY.IMAGE"] = f"docker://{docker_image}"

    delivery_method = si.ask_for_choices(
        "Use Docker or Singularity?",
        choices={"docker": "docker", "singularity": "singularity"},
        default="docker",
    )

    htmap.settings["DELIVERY_METHOD"] = delivery_method
    if delivery_method == "singularity":
        htmap.settings["MAP_OPTIONS.requirements"] = "OpSysMajorVer =?= 7"


def ask_map_options() -> (dict, dict):
    opts = {
        "request_memory": si.ask_for_input("Memory?", default="500MB"),
        "request_disk": si.ask_for_input("Disk?", default="1GB"),
        "max_idle": "100",
    }
    custom_opts = {
        "wantflocking": str(si.ask_for_bool("Want flocking?", default=False)).lower(),
        "wantglidein": str(si.ask_for_bool("Want gliding?", default=False)).lower(),
    }

    return opts, custom_opts
