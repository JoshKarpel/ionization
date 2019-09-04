import datetime

from tqdm import tqdm

import simulacra as si
import simulacra.units as u

import ionization as ion
from . import scan_utils as utils

import htmap


def create_scan(tag):
    parameters = []

    spec_type, mesh_kwargs = utils.ask_mesh_type()
    utils.ask_mask__radial_cosine(parameters, mesh_kwargs)
    utils.ask_numeric_eigenstate_basis(parameters, spec_type=spec_type)

    utils.ask_initial_state_for_hydrogen_sim(parameters)

    utils.ask_mesh_operators(parameters, spec_type=spec_type)
    utils.ask_evolution_method_tdse(parameters)

    utils.ask_time_step(parameters)
    time_initial_in_pw, time_final_in_pw, extra_time = (
        utils.ask_time_evolution_by_pulse_widths()
    )

    # PULSE PARAMETERS
    pulse_parameters = utils.construct_pulses(
        parameters,
        time_initial_in_pw=time_initial_in_pw,
        time_final_in_pw=time_final_in_pw,
    )

    # MISCELLANEOUS
    utils.ask_data_storage_tdse(parameters, spec_type=spec_type)

    # CREATE SPECS
    expanded_parameters = si.expand_parameters(parameters)
    extra_parameters = dict(
        checkpoints=True, checkpoint_every=datetime.timedelta(minutes=20)
    )

    specs = []
    print("Generating specifications...")
    for component, spec_kwargs in enumerate(tqdm(expanded_parameters, ascii=True)):
        electric_potential = spec_kwargs["electric_potential"]

        time_initial = time_initial_in_pw * electric_potential.pulse_width
        time_final = (time_final_in_pw * electric_potential.pulse_width) + extra_time

        spec = spec_type(
            name=component,
            component=component,
            time_initial=time_initial,
            time_final=time_final,
            **mesh_kwargs,
            **spec_kwargs,
            **extra_parameters,
        )

        utils.transfer_potential_attrs_to_spec(electric_potential, spec)

        spec.time_initial_in_pw = time_initial_in_pw
        spec.time_final_in_pw = time_final_in_pw

        specs.append(spec)

    # CREATE MAP
    opts, custom = utils.ask_map_options()

    map = utils.run.map(
        specs, map_options=htmap.MapOptions(**opts, custom_options=custom), tag=tag
    )

    print(f"Created map {map.tag}")

    return map


def main():
    tag = utils.ask_for_tag()
    utils.ask_htmap_settings()

    return create_scan(tag)


if __name__ == "__main__":
    map = main()
