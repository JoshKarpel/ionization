import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
import ionization.ide as ide


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
        target_dir = OUT_DIR,
        # img_format = 'png',
        # fig_dpi_scale = 3,
)

# def comparison_plot(dt_list, t_by_dt, y_by_dt, title):
#     fig = si.vis.get_figure('full')
#     ax = fig.add_subplot(111)
#
#     for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
#         ax.plot(t / asec, np.abs(y) ** 2,
#                 label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
#                 linewidth = .2,
#                 )
#
#     ax.legend(loc = 'best')
#     ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
#     ax.set_ylabel(r'$   \left| a_{\alpha}(t) \right|^2   $')
#     ax.grid(True, **si.vis.GRID_KWARGS)
#
#     si.vis.save_current_figure('{}__comparison'.format(title),
#                                **PLOT_KWARGS)
#
#     plt.close()
#
#
# def error_plot(dt_list, t_by_dt, y_by_dt, title):
#     dt_min_index = np.argmin(dt_list)
#     longest_t = t_by_dt[dt_min_index]
#     best_y = y_by_dt[dt_min_index]
#
#     fig = si.vis.get_figure('full')
#     ax = fig.add_subplot(111)
#
#     for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
#         terp = interp.interp1d(t, y)
#
#         plot_y = terp(longest_t)
#         diff = np.abs(plot_y) ** 2 - np.abs(best_y ** 2)
#
#         ax.plot(longest_t / asec, diff,
#                 label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
#                 linewidth = .2,
#                 )
#
#     ax.legend(loc = 'best')
#     ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
#     ax.set_ylabel(r'$   \left| a_{\alpha}(t) \right|^2 - \left| a_{\alpha}^{\mathrm{best}}(t) \right|^2  $')
#     ax.grid(True, **si.vis.GRID_KWARGS)
#
#     si.vis.save_current_figure('{}__error'.format(title),
#                                **PLOT_KWARGS)
#
#     plt.close()
#
#
# def error_log_plot(dt_list, t_by_dt, y_by_dt, title):
#     dt_min_index = np.argmin(dt_list)
#     longest_t = t_by_dt[dt_min_index]
#     best_y = y_by_dt[dt_min_index]
#
#     fig = si.vis.get_figure('full')
#     ax = fig.add_subplot(111)
#
#     for dt, t, y in zip(dt_list, t_by_dt, y_by_dt):
#         terp = interp.interp1d(t, y)
#
#         plot_y = terp(longest_t)
#         diff = 1 - (np.abs(plot_y) ** 2 / np.abs(best_y ** 2))
#
#         ax.plot(longest_t / asec, diff,
#                 label = r'$\Delta t = {} \, \mathrm{{as}}$'.format(round(dt, 3)),
#                 linewidth = 1,
#                 )
#
#     ax.legend(loc = 'best')
#     ax.set_xlabel(r'Time $t$ ($\mathrm{as}$)')
#     ax.set_ylabel(r'$  1 -  \left| a_{\alpha}(t) \right|^2 / \left| a_{\alpha}^{\mathrm{best}}(t) \right|^2 $')
#
#     ax.grid(True, which = 'major', **si.vis.GRID_KWARGS)
#     ax.grid(True, which = 'minor', **si.vis.GRID_KWARGS)
#
#     ax.set_yscale('log')
#     # ax.set_ylim(bottom = 1e-10, top = 1)
#
#     si.vis.save_current_figure('{}__error_log'.format(title),
#                                **PLOT_KWARGS)
#
#     plt.close()
#
#
# def convergence_plot(dt_list, t_by_dt, y_by_dt, title):
#     dt_min_index = np.argmin(dt_list)
#     longest_t = t_by_dt[dt_min_index]
#     best_y = y_by_dt[dt_min_index]
#
#     fig = si.vis.get_figure('full')
#     ax = fig.add_subplot(111)
#
#     final = [np.abs(np.abs(y[-1]) - np.abs(best_y[-1])) for y in y_by_dt]
#
#     ax.plot(dt_list[:-1], final[:-1])
#
#     ax.set_xlabel(r'Time Step $\Delta t$ ($\mathrm{as}$)')
#     ax.set_ylabel(r'$   \left| \left| a_{\alpha}(t_{\mathrm{final}}) \right| - \left| a_{\alpha}^{\mathrm{best}}(t_{\mathrm{final}}) \right| \right|  $')
#
#     ax.grid(True, which = 'major', **si.vis.GRID_KWARGS)
#     ax.grid(True, which = 'minor', **si.vis.GRID_KWARGS)
#
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     # ax.set_ylim(bottom = .01 * np.nanmin(final), top = 1)
#
#     si.vis.save_current_figure('{}__convergence'.format(title),
#                                **PLOT_KWARGS)
#
#     plt.close()
#
#
# def convergence_plot_squared(dt_list, t_by_dt, y_by_dt, title):
#     dt_min_index = np.argmin(dt_list)
#     longest_t = t_by_dt[dt_min_index]
#     best_y = y_by_dt[dt_min_index]
#
#     fig = si.vis.get_figure('full')
#     ax = fig.add_subplot(111)
#
#     final = [np.abs(np.abs(y[-1]) ** 2 - np.abs(best_y[-1]) ** 2) for y in y_by_dt]
#
#     ax.plot(dt_list[:-1], final[:-1])
#
#     ax.set_xlabel(r'Time Step $\Delta t$ ($\mathrm{as}$)')
#     ax.set_ylabel(r'$   \left| \left| a_{\alpha}(t_{\mathrm{final}}) \right|^2 - \left| a_{\alpha}^{\mathrm{best}}(t_{\mathrm{final}}) \right|^2 \right|  $')
#     ax.grid(True, which = 'major', **si.vis.GRID_KWARGS)
#     ax.grid(True, which = 'minor', **si.vis.GRID_KWARGS)
#
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     # ax.set_ylim(bottom = .01 * np.nanmin(final), top = 1)
#
#     si.vis.save_current_figure('{}__convergence_squared'.format(title),
#                                **PLOT_KWARGS)
#
#     plt.close()

logman = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO)


def run(spec):
    with logman as logger:
        sim = spec.to_sim()

        sim.info().log()

        sim.run()

        sim.plot_b2_vs_time(y_axis_label = r'$   \left| a_{\alpha}(t) \right|^2  $',
                            field_axis_label = r'${}(t)$'.format(ion.LATEX_EFIELD),
                            field_scale = 'AEF',
                            **PLOT_KWARGS)

        sim.info().log()

        return sim


if __name__ == '__main__':
    with logman as logger:
        # electric_field = ion.Rectangle(start_time = -500 * asec, end_time = 500 * asec, amplitude = 1 * atomic_electric_field)

        t_bound_per_pw = 5
        pw = 50

        electric_field = ion.SincPulse(pulse_width = pw * asec, fluence = 1 * Jcm2,
                                       window = ion.RectangularTimeWindow(on_time = -(t_bound_per_pw - 1) * pw * asec,
                                                                          off_time = (t_bound_per_pw - 1) * pw * asec))

        t_bound = pw * t_bound_per_pw

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        # dt_list = np.array([50, 25, 10, 5, 2, 1, .5, .1])
        # dt_list = np.array([10, 5, 2, 1, .5, .1])
        dt_list = np.logspace(1, -2, 10)

        t_by_dt = []
        a_by_dt = []

        # method = 'trapezoid'
        method = 'simpson'

        specs = []
        for dt in dt_list:
            spec = ide.IntegroDifferentialEquationSpecification('{}__{}__dt={}as'.format(method, electric_field.__class__.__name__, round(dt, 3)),
                                                                time_initial = -t_bound * asec, time_final = t_bound * asec, time_step = dt * asec,
                                                                integral_prefactor = prefactor,
                                                                electric_potential = electric_field,
                                                                kernel = ide.gaussian_kernel_LEN, kernel_kwargs = dict(tau_alpha = tau_alpha),
                                                                integration_method = method,
                                                                )

            specs.append(spec)

        results = si.utils.multi_map(run, specs)

        title = '{}__{}'.format(method, electric_field.__class__.__name__)

        si.vis.xxyy_plot(
                title + '__comparison',
                (r.times for r in results),
                (r.b2 for r in results),
                line_labels = (f'${uround(r.spec.time_step, asec, 3)}$ as' for r in results),
                x_label = r'Time $t$', x_unit = 'asec',
                y_label = r'$\left| a(t) \right|^2$',
                **PLOT_KWARGS
        )

        # comparison_plot(dt_list, t_by_dt, a_by_dt, title)
        # error_plot(dt_list, t_by_dt, a_by_dt, title)
        # error_log_plot(dt_list, t_by_dt, a_by_dt, title)
        # convergence_plot(dt_list, t_by_dt, a_by_dt, title)
        # convergence_plot_squared(dt_list, t_by_dt, a_by_dt, title)
