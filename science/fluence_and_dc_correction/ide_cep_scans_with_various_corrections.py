#!/usr/bin/env python
import itertools
import functools
import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u
import simulacra.cluster as clu

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        jp_plain = clu.JobProcessor.load('job_processors/PAPER_OL_ide_cep_scan_plain.job')
        jp_dc = clu.JobProcessor.load('job_processors/PAPER_OL_ide_cep_scan_dc_correct.job')
        jp_dc_and_flu = clu.JobProcessor.load('job_processors/PAPER_OL_ide_cep_scan_dc_and_flu_correct.job')

        jp_to_label = {
            jp_dc: 'dc',
            jp_plain: 'plain',
            jp_dc_and_flu: 'dc and flu',
        }

        for jp in jp_to_label:
            print(jp)

        fluences = np.array(sorted(set.intersection(*[jp.parameter_set('fluence') for jp in jp_to_label])))
        pulse_widths = set.intersection(*[jp.parameter_set('pulse_width') for jp in jp_to_label])
        pulse_widths.remove(400 * u.asec)
        pulse_widths = np.array(sorted(pulse_widths))
        phases = np.array(sorted(set.intersection(*[jp.parameter_set('phase') for jp in jp_to_label])))

        print(fluences / u.Jcm2)
        print(pulse_widths / u.asec)
        print(phases / u.pi)

        for fluence, pulse_width in itertools.product(fluences, pulse_widths):
            jp_to_results = {
                jp: sorted(
                    [r for r in jp.select_by_kwargs(fluence = fluence, pulse_width = pulse_width) if r.pulse_width in pulse_widths],
                    key = lambda x: x.pulse_width
                )
                for jp in jp_to_label
            }

            si.vis.xxyy_plot(
                f'compare__H={u.uround(fluence, u.Jcm2)}jcm2_PW={u.uround(pulse_width, u.asec)}as',
                x_data = [[r.phase for r in results] for jp, results in jp_to_results.items()],
                y_data = [[r.final_initial_state_overlap for r in results] for jp, results in jp_to_results.items()],
                line_labels = [label for jp, label in jp_to_label.items()],
                line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
                x_unit = 'rad',
                title = fr'Comparison $ H = {u.uround(fluence, u.Jcm2)} \, \mathrm{{J/cm^2}}, \tau = {u.uround(pulse_width, u.asec)} \, \mathrm{{as}} $',
                **PLOT_KWARGS
            )

            si.vis.xxyy_plot(
                f'difference__H={u.uround(fluence, u.Jcm2)}jcm2_PW={u.uround(pulse_width, u.asec)}as',
                x_data = [[r.phase for r in results] for jp, results in jp_to_results.items()][1:],
                y_data = [[r.final_initial_state_overlap - plain_r.final_initial_state_overlap
                           for r, plain_r in zip(results, jp_to_results[jp_dc])]
                          for jp, results in jp_to_results.items()][1:],
                line_labels = [label for jp, label in jp_to_label.items()][1:],
                line_kwargs = [{'linestyle': '--'}, {'linestyle': ':'}],
                x_unit = 'rad',
                title = fr'Difference $ H = {u.uround(fluence, u.Jcm2)} \, \mathrm{{J/cm^2}}, \tau = {u.uround(pulse_width, u.asec)} \, \mathrm{{as}} $',
                **PLOT_KWARGS
            )

            si.vis.xxyy_plot(
                f'ratio__H={u.uround(fluence, u.Jcm2)}jcm2_PW={u.uround(pulse_width, u.asec)}as',
                x_data = [[r.phase for r in results] for jp, results in jp_to_results.items()][1:],
                y_data = [[r.final_initial_state_overlap / plain_r.final_initial_state_overlap
                           for r, plain_r in zip(results, jp_to_results[jp_dc])]
                          for jp, results in jp_to_results.items()][1:],
                line_labels = [label for jp, label in jp_to_label.items()][1:],
                line_kwargs = [{'linestyle': '--'}, {'linestyle': ':'}],
                x_unit = 'rad',
                title = fr'Ratio $ H = {u.uround(fluence, u.Jcm2)} \, \mathrm{{J/cm^2}}, \tau = {u.uround(pulse_width, u.asec)} \, \mathrm{{as}} $',
                **PLOT_KWARGS
            )

            si.vis.xxyy_plot(
                f'sym_diff__H={u.uround(fluence, u.Jcm2)}jcm2_PW={u.uround(pulse_width, u.asec)}as',
                x_data = [[r.phase for r in results] for jp, results in jp_to_results.items()][1:],
                y_data = [[(r.final_initial_state_overlap - plain_r.final_initial_state_overlap) / ((r.final_initial_state_overlap + plain_r.final_initial_state_overlap) / 2)
                           for r, plain_r in zip(results, jp_to_results[jp_dc])]
                          for jp, results in jp_to_results.items()][1:],
                line_labels = [label for jp, label in jp_to_label.items()][1:],
                line_kwargs = [{'linestyle': '--'}, {'linestyle': ':'}],
                x_unit = 'rad',
                title = fr'Sym. Diff. $ H = {u.uround(fluence, u.Jcm2)} \, \mathrm{{J/cm^2}}, \tau = {u.uround(pulse_width, u.asec)} \, \mathrm{{as}} $',
                **PLOT_KWARGS
            )

            # si.vis.xxyy_plot(
            #     f'diffs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
            #     x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
            #     y_data = [
            #         [
            #             r.final_initial_state_overlap - br.final_initial_state_overlap
            #             for r, br in zip(results, jp_to_results[original_jp])
            #         ]
            #         for jp, results in jp_to_results.items()
            #     ],
            #     line_labels = [label for jp, label in jp_to_label.items()],
            #     line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
            #     x_unit = 'asec',
            #     **PLOT_KWARGS
            # )

            # si.vis.xxyy_plot(
            #     f'fracs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
            #     x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
            #     y_data = [
            #         [
            #             r.final_initial_state_overlap / br.final_initial_state_overlap
            #             for r, br in zip(results, jp_to_results[original_jp])
            #         ]
            #         for jp, results in jp_to_results.items()
            #     ],
            #     line_labels = [label for jp, label in jp_to_label.items()],
            #     line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
            #     x_unit = 'asec',
            #     **PLOT_KWARGS
            # )
            #
            # si.vis.xxyy_plot(
            #     f'sym_diff_fracs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
            #     x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
            #     y_data = [
            #         [
            #             (r.final_initial_state_overlap - br.final_initial_state_overlap) / ((r.final_initial_state_overlap + br.final_initial_state_overlap) / 2)
            #             for r, br in zip(results, jp_to_results[original_jp])
            #         ]
            #         for jp, results in jp_to_results.items()
            #     ],
            #     line_labels = [label for jp, label in jp_to_label.items()],
            #     line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
            #     x_unit = 'asec',
            #     **PLOT_KWARGS
            # )
