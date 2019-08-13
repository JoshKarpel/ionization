import logging
import os
import itertools

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

T_BOUND_MAP = {ion.SincPulse: 12}
P_BOUND_MAP = {ion.SincPulse: 10}


def make_plot(args):
    pulse_type, pw, flu, cep = args

    t_bound = T_BOUND_MAP[pulse_type]
    p_bound = P_BOUND_MAP[pulse_type]

    times = np.linspace(-t_bound * pw, t_bound * pw, 1e4)

    window = ion.SymmetricExponentialTimeWindow(
        window_time=p_bound * pw, window_width=0.2 * pw
    )
    pulse = pulse_type(pulse_width=pw, fluence=flu, phase=cep, window=window)
    corrected_pulse = ion.DC_correct_electric_potential(pulse, times)

    efield = corrected_pulse.get_electric_field_amplitude(times)
    afield = corrected_pulse.get_vector_potential_amplitude_numeric_cumulative(times)

    starts = range(0, len(times), 50)[50:-50]

    sliced_times = list(times[start:] for start in starts)
    sliced_alphas = list(
        (proton_charge / electron_mass)
        * integ.cumtrapz(
            y=integ.cumtrapz(y=efield[start:], x=times[start:], initial=0),
            x=times[start:],
            initial=0,
        )
        for start in starts
    )

    identifier = f"{pulse_type.__name__}__pw={uround(pw, asec, 0)}as_flu={uround(flu, Jcm2, 2)}jcm2_cep={uround(cep, pi, 2)}pi"

    spec = ion.SphericalHarmonicSpecification(
        identifier,
        r_bound=250 * bohr_radius,
        r_points=250 * 4,
        l_bound=500,
        time_initial=times[0],
        time_final=times[-1],
        use_numeric_eigenstates=True,
        numeric_eigenstate_max_energy=10 * eV,
        numeric_eigenstate_max_angular_momentum=5,
        electric_potential=pulse,
        electric_potential_dc_correction=True,
        mask=ion.RadialCosineMask(
            inner_radius=225 * bohr_radius, outer_radius=250 * bohr_radius
        ),
        checkpoints=True,
        checkpoint_every=50,
        checkpoint_dir=SIM_LIB,
    )

    sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)
    if sim.status != si.Status.FINISHED:
        sim.run_simulation()
        sim.save(target_dir=SIM_LIB)

    si.vis.xxyy_plot(
        identifier,
        [times, times, sim.times, *sliced_times],
        [
            efield / atomic_electric_field,
            afield * (proton_charge / atomic_momentum),
            sim.radial_position_expectation_value_vs_time / bohr_radius,
            *(alpha / bohr_radius for alpha in sliced_alphas),
        ],
        line_labels=[
            rf"$ {ion.LATEX_EFIELD}(t) $",
            rf"$ e \, {ion.LATEX_AFIELD}(t) $",
            rf"$ \left\langle r(t) \right\rangle $",
            rf"$ \alpha(t) $",
        ],
        line_kwargs=[
            None,
            None,
            None,
            *({"color": "black", "alpha": 0.5} for _ in starts),
        ],
        x_label=r"Time $t$",
        x_unit="asec",
        # y_label = r'Field Amplitude (a.u.) / Distance ($a_0$)',
        title=rf"$ \tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \; H = {uround(flu, Jcm2, 2)} \, \mathrm{{J/cm^2}}, \; \varphi = {uround(cep, pi, 2)}\pi $",
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with logman as logger:
        pulse_widths = np.array([200, 400, 800, 2000]) * asec
        fluences = np.array([0.01, 0.1, 1, 2, 5, 10]) * Jcm2
        phases = [0, pi / 4, pi / 2]
        pulse_types = [ion.SincPulse, ion.GaussianPulse]

        si.utils.multi_map(
            make_plot,
            list(itertools.product(pulse_types, pulse_widths, fluences, phases)),
            processes=4,
        )
