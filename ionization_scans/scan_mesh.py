from pathlib import Path

from tqdm import tqdm

import scipy.optimize as opt

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import ionization as ion
from . import mesh_scan_utils as msu

import htmap


def create_scan(tag):
    parameters = []

    spec_type, mesh_kwargs = msu.ask_mesh_type()
    msu.ask_mask__radial_cosine(parameters, mesh_kwargs)
    msu.ask_numeric_eigenstate_basis(parameters, spec_type=spec_type)

    msu.ask_initial_state_for_hydrogen_sim(parameters)

    msu.ask_mesh_operators(parameters, spec_type=spec_type)
    msu.ask_evolution_method_tdse(parameters)

    msu.ask_time_step(parameters)
    time_initial_in_pw, time_final_in_pw, extra_time = (
        msu.ask_time_evolution_by_pulse_widths()
    )

    # PULSE PARAMETERS
    pulse_parameters = msu.construct_pulses(
        parameters,
        time_initial_in_pw=time_initial_in_pw,
        time_final_in_pw=time_final_in_pw,
    )

    # MISCELLANEOUS
    msu.ask_data_storage_tdse(parameters, spec_type=spec_type)

    # CREATE SPECS
    expanded_parameters = si.cluster.expand_parameters(parameters)
    extra_parameters = {}

    final_parameters = [
        dict(component=c, **params, **extra_parameters)
        for c, params in enumerate(expanded_parameters)
    ]

    specs = []
    print("Generating specifications...")
    for job_number, spec_kwargs in enumerate(tqdm(final_parameters, ascii=True)):
        electric_potential = spec_kwargs["electric_potential"]

        time_initial = time_initial_in_pw * electric_potential.pulse_width
        time_final = (time_final_in_pw * electric_potential.pulse_width) + extra_time

        spec = spec_type(
            name=job_number,
            time_initial=time_initial,
            time_final=time_final,
            **mesh_kwargs,
            **spec_kwargs,
        )

        msu.transfer_potential_attrs_to_spec(electric_potential, spec)

        spec.time_initial_in_pw = time_initial_in_pw
        spec.time_final_in_pw = time_final_in_pw

        specs.append(spec)

    # CREATE MAP
    opts, custom = msu.ask_map_options()

    map = msu.run.map(
        specs, map_options=htmap.MapOptions(**opts, custom_options=custom), tag=tag
    )

    print(f"Created map {map.tag}")

    return map


def main():
    tag = msu.ask_for_tag()
    msu.ask_htmap_settings()

    return create_scan(tag)


if __name__ == "__main__":
    map = main()
