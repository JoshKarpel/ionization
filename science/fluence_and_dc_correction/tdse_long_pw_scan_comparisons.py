#!/usr/bin/env python

import logging
import os
import functools

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
        original_jp = clu.JobProcessor.load('job_processors/PAPER_tdse_pw_scans.job')
        longer_jp = clu.JobProcessor.load('job_processors/PAPER__tdse_pw_scan_longer.job')
        longer_no_dc_jp = clu.JobProcessor.load('job_processors/PAPER__tdse_pw_scan_longer_no_DC_correct.job')

        jp_to_label = {
            original_jp: 'original',
            longer_jp: 'longer',
            longer_no_dc_jp: 'longer_no_dc',
        }

        for jp in jp_to_label:
            print(jp)

        fluences = set.intersection(*[jp.parameter_set('fluence') for jp in jp_to_label])
        print(fluences)
        pulse_widths = set.intersection(*[jp.parameter_set('pulse_width') for jp in jp_to_label])

        for fluence in fluences:
            for phase in (0, u.pi / 2):
                jp_to_results = {
                    jp: sorted(
                        # jp.select_by_kwargs(fluence = fluence, phase = phase),
                        [r for r in jp.select_by_kwargs(fluence = fluence, phase = phase) if r.pulse_width in pulse_widths],
                        key = lambda x: x.pulse_width
                    )
                    for jp in jp_to_label
                }

                si.vis.xxyy_plot(
                    f'compare__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
                    x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
                    y_data = [[r.final_initial_state_overlap for r in results] for jp, results in jp_to_results.items()],
                    line_labels = [label for jp, label in jp_to_label.items()],
                    line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
                    x_unit = 'asec',
                    **PLOT_KWARGS
                )

                si.vis.xxyy_plot(
                    f'diffs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
                    x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
                    y_data = [
                        [
                            r.final_initial_state_overlap - br.final_initial_state_overlap
                            for r, br in zip(results, jp_to_results[original_jp])
                        ]
                        for jp, results in jp_to_results.items()
                    ],
                    line_labels = [label for jp, label in jp_to_label.items()],
                    line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
                    x_unit = 'asec',
                    **PLOT_KWARGS
                )

                si.vis.xxyy_plot(
                    f'fracs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
                    x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
                    y_data = [
                        [
                            r.final_initial_state_overlap / br.final_initial_state_overlap
                            for r, br in zip(results, jp_to_results[original_jp])
                        ]
                        for jp, results in jp_to_results.items()
                    ],
                    line_labels = [label for jp, label in jp_to_label.items()],
                    line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
                    x_unit = 'asec',
                    **PLOT_KWARGS
                )

                si.vis.xxyy_plot(
                    f'sym_diff_fracs__{u.uround(fluence, u.Jcm2)}jcm2_{u.uround(phase, u.pi)}pi',
                    x_data = [[r.pulse_width for r in results if r.pulse_width] for jp, results in jp_to_results.items()],
                    y_data = [
                        [
                            (r.final_initial_state_overlap - br.final_initial_state_overlap) / ((r.final_initial_state_overlap + br.final_initial_state_overlap) / 2)
                            for r, br in zip(results, jp_to_results[original_jp])
                        ]
                        for jp, results in jp_to_results.items()
                    ],
                    line_labels = [label for jp, label in jp_to_label.items()],
                    line_kwargs = [None, {'linestyle': '--'}, {'linestyle': ':'}],
                    x_unit = 'asec',
                    **PLOT_KWARGS
                )
