import os
import itertools

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)


def get_amplitude(sim_result):
    return sim_result.electric_potential[0].amplitude_time


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        jp_name = 'hyd__amp_scan__test3_no_flockglide.job'
        jp = iclu.PulseJobProcessor.load(jp_name)

        pulse_widths = jp.parameter_set('pulse_width')
        phases = jp.parameter_set('phase')
        amplitudes = np.array(sorted(set(get_amplitude(r) for r in jp.data.values())))

        print(pulse_widths)
        print(phases)
        print(amplitudes / atomic_electric_field)

        metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
        metric_names = ['Initial State Overlap', 'Bound State Overlap']
        for metric, metric_name in zip(metrics, metric_names):
            for amplitude in amplitudes:
                results = [r for r in jp.data.values() if get_amplitude(r) == amplitude]

                si.vis.xxyy_plot(
                    f'{metric}__amp={uround(amplitude, atomic_electric_field)}aef',
                    [
                        *[[r.pulse_width for r in results if r.phase == phase] for phase in phases],
                    ],
                    [
                        *[[getattr(r, metric) for r in results if r.phase == phase] for phase in phases],
                    ],
                    line_labels = [fr'$\varphi = {uround(phase, pi)}\pi$' for phase in phases],
                    x_label = r'$\tau$', x_unit = 'asec',
                    y_label = metric_name,
                    title = fr'Constant Amplitude Scan: ${ion.LATEX_EFIELD} = {uround(amplitude, atomic_electric_field)} \, \mathrm{{a.u.}}$',
                    **PLOT_KWARGS,
                )
