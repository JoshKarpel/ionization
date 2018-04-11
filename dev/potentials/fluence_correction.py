#!/usr/bin/env python
import datetime
import logging
import os

import numpy as np
import numpy.fft as nfft

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run_from_simlib(spec):
    with LOGMAN:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()

        return sim


def fft_field(field, times):
    dt = np.abs(times[1] - times[0])
    freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
    df = np.abs(freqs[1] - freqs[0])
    fft = tuple(nfft.fftshift(nfft.fft(nfft.fftshift(field), norm = 'ortho') / df))

    return freqs, fft


if __name__ == '__main__':
    with LOGMAN as logger:
        pw = 100 * u.asec
        tw = 30 * pw
        tb = 35 * pw

        flu = 1 * u.Jcm2

        window = ion.potentials.SymmetricExponentialTimeWindow(
            window_time = tw,
            window_width = pw / 5,
        )

        uncorrected_cos_pulse = ion.potentials.SincPulse(
            pulse_width = pw,
            phase = 0,
            fluence = flu,
            window = window,
        )
        uncorrected_sin_pulse = ion.potentials.SincPulse(
            pulse_width = pw,
            phase = u.pi / 2,
            fluence = flu,
            window = window,
        )

        times = np.linspace(-tb, tb, int(2 * tb / u.asec))
        dt = np.abs(times[1] - times[0])

        dc_corrected_cos_pulse = ion.potentials.DC_correct_electric_potential(uncorrected_cos_pulse, times)
        dc_corrected_sin_pulse = ion.potentials.DC_correct_electric_potential(uncorrected_sin_pulse, times)
        # dc_corrected_cos_pulse = uncorrected_cos_pulse
        # dc_corrected_sin_pulse = uncorrected_sin_pulse

        # print(dc_corrected_cos_pulse.info())

        cos_integral_of_field = dc_corrected_cos_pulse.get_electric_field_integral_numeric(times)
        sin_integral_of_field = dc_corrected_sin_pulse.get_electric_field_integral_numeric(times)
        print(f'cos field integral: {u.uround(cos_integral_of_field, u.atomic_electric_field * u.asec, 15)}')
        print(f'sin field integral: {u.uround(sin_integral_of_field, u.atomic_electric_field * u.asec, 15)}')

        cos_fluence = dc_corrected_cos_pulse.get_fluence_numeric(times)
        sin_fluence = dc_corrected_sin_pulse.get_fluence_numeric(times)
        print(f'cos fluence: {u.uround(cos_fluence, u.Jcm2, 15)}')
        print(f'sin fluence: {u.uround(sin_fluence, u.Jcm2, 15)}')

        fluence_and_dc_corrected_cos_pulse = ion.potentials.FluenceCorrector(
            electric_potential = dc_corrected_cos_pulse,
            times = times,
            target_fluence = flu,
        )
        fluence_and_dc_corrected_sin_pulse = ion.potentials.FluenceCorrector(
            electric_potential = dc_corrected_sin_pulse,
            times = times,
            target_fluence = flu,
        )

        print('\nAFTER FLUENCE CORRECTION' + '\n' + '-' * 80 + '\n')

        corrected_cos_integral_of_field = fluence_and_dc_corrected_cos_pulse.get_electric_field_integral_numeric(times)
        corrected_sin_integral_of_field = fluence_and_dc_corrected_sin_pulse.get_electric_field_integral_numeric(times)
        print(f'cos field integral: {u.uround(corrected_cos_integral_of_field, u.atomic_electric_field * u.atomic_time, 15)}')
        print(f'sin field integral: {u.uround(corrected_sin_integral_of_field, u.atomic_electric_field * u.atomic_time, 15)}')

        corrected_cos_fluence = fluence_and_dc_corrected_cos_pulse.get_fluence_numeric(times)
        corrected_sin_fluence = fluence_and_dc_corrected_sin_pulse.get_fluence_numeric(times)
        print(f'cos fluence: {u.uround(corrected_cos_fluence, u.Jcm2, 15)}')
        print(f'sin fluence: {u.uround(corrected_sin_fluence, u.Jcm2, 15)}')

        print(f'cos correction ratio: {fluence_and_dc_corrected_cos_pulse.amplitude_correction_ratio}')
        print(f'sin correction ratio: {fluence_and_dc_corrected_sin_pulse.amplitude_correction_ratio}')

        si.vis.xy_plot(
            'fields_vs_time',
            times,
            dc_corrected_cos_pulse.get_electric_field_amplitude(times),
            dc_corrected_sin_pulse.get_electric_field_amplitude(times),
            fluence_and_dc_corrected_cos_pulse.get_electric_field_amplitude(times),
            fluence_and_dc_corrected_sin_pulse.get_electric_field_amplitude(times),
            line_labels = [
                'cos',
                'sin',
                'cos corrected',
                'sin corrected',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': '--'},
            ],
            x_unit = 'asec',
            y_unit = 'atomic_electric_field',
            **PLOT_KWARGS
        )

        si.vis.xy_plot(
            'fields_vs_time__zom',
            times,
            dc_corrected_cos_pulse.get_electric_field_amplitude(times),
            dc_corrected_sin_pulse.get_electric_field_amplitude(times),
            fluence_and_dc_corrected_cos_pulse.get_electric_field_amplitude(times),
            fluence_and_dc_corrected_sin_pulse.get_electric_field_amplitude(times),
            line_labels = [
                'cos',
                'sin',
                'cos corrected',
                'sin corrected',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': '--'},
            ],
            x_unit = 'asec',
            y_unit = 'atomic_electric_field',
            x_lower_limit = -5 * pw,
            x_upper_limit = 5 * pw,
            **PLOT_KWARGS
        )

        cos_freq, cos_fft = fft_field(dc_corrected_cos_pulse.get_electric_field_amplitude(times), times)
        sin_freq, sin_fft = fft_field(dc_corrected_sin_pulse.get_electric_field_amplitude(times), times)
        cos_power = np.abs(cos_fft) ** 2
        sin_power = np.abs(sin_fft) ** 2

        corrected_cos_freq, corrected_cos_fft = fft_field(fluence_and_dc_corrected_cos_pulse.get_electric_field_amplitude(times), times)
        corrected_sin_freq, corrected_sin_fft = fft_field(fluence_and_dc_corrected_sin_pulse.get_electric_field_amplitude(times), times)
        corrected_cos_power = np.abs(corrected_cos_fft) ** 2
        corrected_sin_power = np.abs(corrected_sin_fft) ** 2

        f_max = 11000 * u.THz
        si.vis.xxyy_plot(
            'power_vs_freq',
            [
                cos_freq,
                sin_freq,
                corrected_cos_freq,
                corrected_sin_freq,
            ],
            [
                cos_power,
                sin_power,
                corrected_cos_power,
                corrected_sin_power,
            ],
            line_labels = [
                'cos',
                'sin',
                'cos corrected',
                'sin corrected',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': '--'},
            ],
            x_unit = 'THz',
            x_lower_limit = -f_max,
            x_upper_limit = f_max,
            **PLOT_KWARGS,
        )

        f_max = 2000 * u.THz
        si.vis.xxyy_plot(
            'power_vs_freq_zoom',
            [
                cos_freq,
                sin_freq,
                corrected_cos_freq,
                corrected_sin_freq,
            ],
            [
                cos_power,
                sin_power,
                corrected_cos_power,
                corrected_sin_power,
            ],
            line_labels = [
                'cos',
                'sin',
                'cos corrected',
                'sin corrected',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': '--'},
            ],
            x_unit = 'THz',
            x_lower_limit = -f_max,
            x_upper_limit = f_max,
            **PLOT_KWARGS,
        )

        spec_kwargs = dict(
            time_initial = -tb,
            time_final = tb,
            time_step = dt,
            r_bound = 100 * u.bohr_radius,
            r_points = 100 * 20,
            l_bound = 500,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * u.eV,
            numeric_eigenstate_max_angular_momentum = 10,
            store_data_every = 1,
            checkpoints = True,
            checkpoint_dir = SIM_LIB,
            checkpoint_every = datetime.timedelta(minutes = 1),
        )

        specs = [
            ion.SphericalHarmonicSpecification(
                'uncorrected_cos',
                electric_potential = uncorrected_cos_pulse,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'uncorrected_sin',
                electric_potential = uncorrected_sin_pulse,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'dc_corrected_cos',
                electric_potential = dc_corrected_cos_pulse,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'dc_corrected_sin',
                electric_potential = dc_corrected_sin_pulse,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'fluence_and_dc_corrected_cos',
                electric_potential = fluence_and_dc_corrected_cos_pulse,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'fluence_and_dc_corrected_sin',
                electric_potential = fluence_and_dc_corrected_sin_pulse,
                **spec_kwargs,
            )
        ]

        # for spec in specs:
        #     print()
        #     print(spec.info())
        #     print()

        # sims = si.utils.multi_map(run_from_simlib, specs, processes = 2)

        # for sim in sims:
        #     sim.plot_wavefunction_vs_time(**PLOT_KWARGS)
