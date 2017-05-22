import logging

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import simulacra as si

from simulacra.units import *

from . import core


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COLORMESH_GRID_KWARGS = {
    **si.plots.COLORMESH_GRID_KWARGS,
    **dict(
            linestyle = ':',
            linewidth = 1.5,
            alpha = 0.6
    )
}


# class MetricsAndElectricField(si.AxisManager):
#     def __init__(self,
#                  log_metrics = False,
#                  time_unit = 'asec',
#                  electric_field_unit = 'AEF',
#                  metrics = ('norm',),
#                  label_top = False,
#                  show_y_label = True,
#                  ticks_top = False,
#                  legend_kwargs = None):
#         self.time_unit_str = ''
#         if type(time_unit) == str:
#             self.time_unit_str = UNIT_NAME_TO_LATEX[time_unit]
#             time_unit = UNIT_NAME_TO_VALUE[time_unit]
#         self.time_unit = time_unit
#
#         if type(electric_field_unit) == str:
#             self.electric_field_unit_str = UNIT_NAME_TO_LATEX[electric_field_unit]
#             self.electric_field_unit = UNIT_NAME_TO_VALUE[electric_field_unit]
#         else:
#             self.electric_field_unit_str = ''
#             self.electric_field_unit = electric_field_unit
#
#         self.log_metrics = log_metrics
#
#         self.label_top = label_top
#         self.show_y_label = show_y_label
#         self.ticks_top = ticks_top
#
#         if legend_kwargs is None:
#             legend_kwargs = dict()
#         legend_defaults = dict(
#                 loc = 'lower left',
#                 fontsize = 20,
#                 fancybox = True,
#                 framealpha = .1,
#         )
#         self.legend_kwargs = {**legend_defaults, **legend_kwargs}
#
#         self.metrics = metrics
#
#         super(MetricsAndElectricField, self).__init__()
#
#     def initialize_axis(self):
#         self.time_line, = self.axis.plot([self.sim.data_times[self.sim.data_time_index] / self.time_unit,
#                                           self.sim.data_times[self.sim.data_time_index] / self.time_unit],
#                                          [0, 2],
#                                          color = 'gray',
#                                          animated = True)
#
#         self.redraw += [self.time_line]
#
#         self._initialize_electric_field()
#
#         for metric in self.metrics:
#             self.__getattribute__('_initialize_metric_' + metric)()
#
#         self.legend = self.axis.legend(**self.legend_kwargs)
#         self.redraw += [self.legend]
#
#         self.axis.grid(True, color = 'gray', linestyle = '--')
#
#         self.axis.set_xlabel(r'Time $t$ (${}$)'.format(self.time_unit_str), fontsize = 24)
#
#         if self.show_y_label:
#             self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)
#
#         if self.label_top:
#             self.axis.xaxis.set_label_position('top')
#
#         self.axis.tick_params(labeltop = self.ticks_top)
#
#         self.axis.set_xlim(self.sim.data_times[0] / self.time_unit, self.sim.data_times[-1] / self.time_unit)
#         if self.log_metrics:
#             self.axis.set_yscale('log')
#             self.axis.set_ylim(1e-8, 1)
#         else:
#             self.axis.set_ylim(0, 1.025)
#         self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)
#
#         self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]
#
#         super().initialize_axis()
#
#     def _initialize_electric_field(self):
#         self.axis_field = self.axis.twinx()
#
#         y_limit = 1.05 * np.nanmax(np.abs(self.spec.electric_potential.get_electric_field_amplitude(self.sim.data_times))) / self.electric_field_unit
#         self.axis_field.set_ylim(-y_limit, y_limit)
#
#         self.axis_field.set_ylabel(r'${}(t)$ (${}$)'.format(core.LATEX_EFIELD, self.electric_field_unit_str),
#                                    fontsize = 24, color = '#d62728')
#         self.axis_field.yaxis.set_label_position('right')
#         self.axis_field.tick_params(axis = 'both', which = 'major', labelsize = 14)
#         self.axis_field.grid(True, color = '#d62728', linestyle = ':')
#
#         for tick in self.axis_field.get_yticklabels():
#             tick.set_color('#d62728')
#
#         self.electric_field_line, = self.axis_field.plot(self.sim.data_times / self.time_unit,
#                                                          self.sim.electric_field_amplitude_vs_time / self.electric_field_unit,
#                                                          label = r'$E(t)$ ({})'.format(self.electric_field_unit_str),
#                                                          color = core.COLOR_ELECTRIC_FIELD, linewidth = 3,
#                                                          animated = True)
#
#         self.redraw += [self.electric_field_line, *self.axis_field.xaxis.get_gridlines(), *self.axis_field.yaxis.get_gridlines()]
#
#     def _initialize_metric_norm(self):
#         self.norm_line, = self.axis.plot(self.sim.data_times / self.time_unit,
#                                          self.sim.norm_vs_time,
#                                          label = r'$\left\langle \psi|\psi \right\rangle$',
#                                          color = 'black', linewidth = 3,
#                                          animated = True)
#
#         self.redraw += [self.norm_line]
#
#     def _initialize_metric_initial_state_overlap(self):
#         self.initial_state_overlap_line, = self.axis.plot(self.sim.data_times / self.time_unit,
#                                                           self.sim.state_overlaps_vs_time[self.sim.spec.initial_state],
#                                                           label = r'$\left| \left\langle \psi|{} \right\rangle \right|^2$'.format(self.sim.spec.initial_state.latex),
#                                                           color = 'blue', linewidth = '3',
#                                                           animated = True)
#
#         self.redraw += [self.initial_state_overlap_line]
#
#     def update_axis(self):
#         self._update_electric_field()
#
#         for metric in self.metrics:
#             self.__getattribute__('_update_metric_' + metric)()
#
#         self.time_line.set_xdata([self.sim.data_times[self.sim.data_time_index] / self.time_unit, self.sim.data_times[self.sim.data_time_index] / self.time_unit])
#
#         super().update_axis()
#
#     def _update_electric_field(self):
#         self.electric_field_line.set_ydata(self.sim.electric_field_amplitude_vs_time / self.electric_field_unit)
#
#     def _update_metric_norm(self):
#         self.norm_line.set_ydata(self.sim.norm_vs_time)
#
#     def _update_metric_initial_state_overlap(self):
#         self.initial_state_overlap_line.set_ydata(self.sim.state_overlaps_vs_time[self.sim.spec.initial_state])

class ElectricPotentialAxis(si.AxisManager):
    def __init__(self,
                 time_unit = 'asec',
                 show_electric_field = True,
                 electric_field_unit = 'AEF',
                 show_vector_potential = False,
                 vector_potential_unit = 'atomic_momentum',
                 show_y_label = False,
                 show_ticks_bottom = True,
                 show_ticks_top = False,
                 show_ticks_right = True,
                 show_ticks_left = True,
                 legend_kwargs = None):
        self.show_electric_field = show_electric_field
        self.show_vector_potential = show_vector_potential

        if not show_electric_field and not show_vector_potential:
            logger.warning(f'{self} has both show_electric_field and show_vector_potential set to False')

        self.time_unit = time_unit
        self.time_unit_value, self.time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)
        self.electric_field_unit = electric_field_unit
        self.electric_field_unit_value, self.electric_field_unit_latex = get_unit_value_and_latex_from_unit(electric_field_unit)
        self.vector_potential_unit = vector_potential_unit
        self.vector_potential_unit_value, self.vector_potential_unit_latex = get_unit_value_and_latex_from_unit(vector_potential_unit)

        self.show_y_label = show_y_label
        self.show_ticks_bottom = show_ticks_bottom
        self.show_ticks_top = show_ticks_top
        self.show_ticks_right = show_ticks_right
        self.show_ticks_left = show_ticks_left

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
                loc = 'lower left',
                fontsize = 20,
                fancybox = True,
                framealpha = .1,
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        super().__init__()

    def initialize_axis(self):
        t = self.sim.time / self.time_unit_value
        self.time_line, = self.axis.plot([t, t],
                                         [0, 2],
                                         color = 'gray',
                                         animated = True)
        self.redraw.append(self.time_line)

        if self.show_electric_field:
            self.electric_field_line, = self.axis.plot(self.sim.data_times / self.time_unit_value,
                                                       self.sim.electric_field_amplitude_vs_time / self.electric_field_unit_value,
                                                       # label = fr'${core.LATEX_EFIELD}(t)$ (${self.electric_field_unit_latex}$)',
                                                       label = fr'${core.LATEX_EFIELD}(t)$',
                                                       color = core.COLOR_ELECTRIC_FIELD, linewidth = 2,
                                                       animated = True)

            self.redraw.append(self.electric_field_line)

        if self.show_vector_potential:
            self.vector_potential_line, = self.axis.plot(self.sim.data_times / self.time_unit_value,
                                                         proton_charge * self.sim.vector_potential_amplitude_vs_time / self.vector_potential_unit_value,
                                                         # label = fr'$e{core.LATEX_AFIELD}(t)$ (${self.vector_potential_unit_latex}$)',
                                                         label = fr'$e{core.LATEX_AFIELD}(t)$',
                                                         color = core.COLOR_VECTOR_POTENTIAL, linewidth = 2,
                                                         animated = True)

            self.redraw.append(self.vector_potential_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **si.plots.GRID_KWARGS)

        self.axis.set_xlabel(fr'Time $t$ (${self.time_unit_latex}$)', fontsize = 24)

        if self.show_y_label:
            self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)

        self.axis.tick_params(labelbottom = self.show_ticks_bottom, labeltop = self.show_ticks_top, labelright = self.show_ticks_right, labelleft = self.show_ticks_left)

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit_value, self.sim.data_times[-1] / self.time_unit_value)

        data = []
        if self.show_electric_field:
            data.append(self.spec.electric_potential.get_electric_field_amplitude(self.sim.times) / self.electric_field_unit_value)
        if self.show_vector_potential:
            data.append(proton_charge * self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times) / self.vector_potential_unit_value)

        y_lower_limit, y_upper_limit = si.plots.set_axis_limits(self.axis,
                                                                *data,
                                                                pad = 0.05,
                                                                direction = 'y',
                                                                )

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def update_axis(self):
        if self.show_electric_field:
            self.electric_field_line.set_ydata(self.sim.electric_field_amplitude_vs_time / self.electric_field_unit_value)
        if self.show_vector_potential:
            self.vector_potential_line.set_ydata(proton_charge * self.sim.vector_potential_amplitude_vs_time / self.vector_potential_unit_value)

        t = self.sim.time / self.time_unit_value
        self.time_line.set_xdata([t, t])

        super().update_axis()


class TestStateStackplot(si.AxisManager):
    def __init__(self,
                 states = None,
                 show_norm = True,
                 time_unit = 'asec',
                 show_y_label = False,
                 show_ticks_bottom = True,
                 show_ticks_top = False,
                 show_ticks_right = True,
                 show_ticks_left = True,
                 legend_kwargs = None):
        self.states = states
        self.show_norm = show_norm

        self.time_unit = time_unit
        self.time_unit_value, self.time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

        self.show_y_label = show_y_label
        self.show_ticks_bottom = show_ticks_bottom
        self.show_ticks_top = show_ticks_top
        self.show_ticks_right = show_ticks_right
        self.show_ticks_left = show_ticks_left

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
                loc = 'lower left',
                fontsize = 20,
                fancybox = True,
                framealpha = .1,
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        super().__init__()

    def initialize_axis(self):
        if self.show_norm:
            self.norm_line, = self.axis.plot(self.sim.data_times / self.time_unit_value,
                                             self.sim.norm_vs_time,
                                             label = r'$\left\langle \psi|\psi \right\rangle$',
                                             color = 'black',
                                             linewidth = 2)

            self.redraw.append(self.norm_line)

        self._initialize_stackplot()

        t = self.sim.time / self.time_unit_value
        self.time_line, = self.axis.plot([t, t],
                                         [0, 2],
                                         color = 'gray',
                                         animated = True)
        self.redraw.append(self.time_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **si.plots.GRID_KWARGS)

        self.axis.set_xlabel(fr'Time $t$ (${self.time_unit_latex}$)', fontsize = 24)

        if self.show_y_label:
            self.axis.set_ylabel('Wavefunction Metric', fontsize = 24)

        self.axis.tick_params(labelbottom = self.show_ticks_bottom, labeltop = self.show_ticks_top, labelright = self.show_ticks_right, labelleft = self.show_ticks_left)

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit_value, self.sim.data_times[-1] / self.time_unit_value)
        self.axis.set_ylim(0, 1)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 14)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def _get_stackplot_data(self):
        # return [self.sim.state_overlaps_vs_time[state] for state in self.spec.test_states]

        state_overlaps = self.sim.state_overlaps_vs_time
        if self.states is not None:
            if callable(self.states):
                state_overlaps = {state: overlap for state, overlap in state_overlaps.items() if self.states(state)}
            else:
                states = set(self.states)
                state_overlaps = {state: overlap for state, overlap in state_overlaps.items() if state in states or (state.numeric and state.analytic_state in states)}

        return state_overlaps

    def _initialize_stackplot(self):
        stackplot_data = self._get_stackplot_data()

        self.overlaps_stackplot = self.axis.stackplot(self.sim.data_times / self.time_unit_value,
                                                      *stackplot_data.values(),
                                                      labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.latex) for state in stackplot_data.keys()],
                                                      animated = True)

        self.redraw += [*self.overlaps_stackplot]

    def _update_stackplot_lines(self):
        for x in self.overlaps_stackplot:
            self.redraw.remove(x)
            x.remove()

        self.axis.set_color_cycle(None)

        stackplot_data = self._get_stackplot_data()

        self.overlaps_stackplot = self.axis.stackplot(self.sim.data_times / self.time_unit_value,
                                                      *stackplot_data.values(),
                                                      labels = [r'$\left| \left\langle \psi| {} \right\rangle \right|^2$'.format(state.latex) for state in stackplot_data.keys()],
                                                      animated = True)

        self.redraw = [*self.overlaps_stackplot] + self.redraw

    def update_axis(self):
        if self.show_norm:
            self.norm_line.set_ydata(self.sim.norm_vs_time)

        self._update_stackplot_lines()

        t = self.sim.time / self.time_unit_value
        self.time_line.set_xdata([t, t])

        super().update_axis()


class QuantumMeshAxis(si.AxisManager):
    def __init__(self,
                 which = 'g2',
                 colormap = core.COLORMAP_WAVEFUNCTION,
                 norm = si.plots.AbsoluteRenormalize(),
                 plot_limit = None,
                 distance_unit = 'bohr_radius',
                 shading = 'gouraud',
                 slicer = 'get_mesh_slicer'):
        self.which = which
        self.colormap = colormap
        self.norm = norm
        self.plot_limit = plot_limit
        self.distance_unit = distance_unit
        self.shading = shading
        self.slicer = slicer

        super().__init__()

    def initialize(self, simulation):
        self.attach_method = getattr(simulation.mesh, f'attach_{self.which}_to_axis')
        self.update_method = getattr(simulation.mesh, f'update_{self.which}_mesh')

        super().initialize(simulation)

    # def initialize_axis(self):
    #     # self.mesh = self.attach_method(self.axis,
    #     #                                colormap = self.colormap,
    #     #                                norm = self.norm,
    #     #                                shading = self.shading,
    #     #                                plot_limit = self.plot_limit,
    #     #                                distance_unit = self.distance_unit,
    #     #                                slicer = self.slicer,
    #     #                                animated = True)
    #     # self.redraw.append(self.mesh)
    #
    #     #  doesn't work if its here....
    #
    #     super().initialize_axis()

    def update_axis(self):
        self.update_method(self.mesh,
                           shading = self.shading,
                           plot_limit = self.plot_limit,
                           slicer = self.slicer)

        super().update_axis()


class WavefunctionSimulationAnimator(si.Animator):
    def __init__(self,
                 axman_wavefunction = None,
                 **kwargs):
        super(WavefunctionSimulationAnimator, self).__init__(**kwargs)

        self.axman_wavefunction = axman_wavefunction

    def __str__(self):
        return si.utils.field_str(self, 'postfix', 'axis_managers')

    def __repr__(self):
        return self.__str__()


# class LineMeshAxis(QuantumMeshAxis):
#     def initialize_axis(self):
#         unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)
#
#         self.mesh = self.sim.mesh.attach_g2_to_axis(self.axis, plot_limit = self.plot_limit, distance_unit = self.distance_unit, animated = True)
#         self.redraw += [self.mesh]
#
#         self.axis.grid(True, color = si.plots.COLOR_OPPOSITE_INFERNO, linestyle = ':')  # change grid color to make it show up against the colormesh
#
#         self.axis.set_xlabel(r'$x$ (${}$)'.format(unit_name), fontsize = 24)
#         self.axis.set_ylabel(r'$\left|\psi\right|^2$', fontsize = 30)
#
#         self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)
#         self.axis.tick_params(labelright = True, labeltop = True)
#
#         self.axis.axis('tight')
#
#         self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)
#
#         super(LineMeshAxis, self).initialize()
#
#     def update_axis(self):
#         self.sim.mesh.update_g2_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)
#
#         super().update_axis()


# class LineAnimator(WavefunctionSimulationAnimator):
#     def _initialize_figure(self):
#         self.fig = plt.figure(figsize = (16, 12))
#
#         self.ax_mesh = LineMeshAxis(self.fig.add_axes([.07, .34, .88, .62]), self.sim,
#                                     plot_limit = self.plot_limit,
#                                     renormalize = self.renormalize,
#                                     log_g = self.log_g,
#                                     overlay_probability_current = self.overlay_probability_current,
#                                     distance_unit = self.distance_unit)
#         self.ax_metrics = MetricsAndElectricField(self.fig.add_axes([.065, .065, .85, .2]), self.sim,
#                                                   log_metrics = self.log_metrics,
#                                                   metrics = self.metrics)
#
#         self.axis_managers += [self.ax_mesh, self.ax_metrics]
#
#         super(LineAnimator, self)._initialize_figure()


class CylindricalSliceMeshAxis(QuantumMeshAxis):
    def initialize_axis(self):
        unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)

        if self.which == 'g':
            self.norm.equator_magnitude = np.max(np.abs(self.sim.mesh.g) / core.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)

        self.mesh = self.attach_method(self.axis,
                                       colormap = self.colormap,
                                       norm = self.norm,
                                       shading = self.shading,
                                       plot_limit = self.plot_limit,
                                       distance_unit = self.distance_unit,
                                       slicer = self.slicer,
                                       animated = True)
        self.redraw.append(self.mesh)

        self.axis.grid(True, color = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], **COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r'$z$ (${}$)'.format(unit_name), fontsize = 24)
        self.axis.set_ylabel(r'$\rho$ (${}$)'.format(unit_name), fontsize = 24)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)

        self.axis.axis('tight')

        super().initialize_axis()

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(), *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        if self.which != 'g':
            divider = make_axes_locatable(self.axis)
            cax = divider.append_axes("right", size = "2%", pad = 0.05)
            self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
            self.cbar.ax.tick_params(labelsize = 20)


class CylindricalSliceAnimator(WavefunctionSimulationAnimator):
    def __init__(self,
                 axman_lower = ElectricPotentialAxis(),
                 **kwargs):
        self.axman_lower = axman_lower

        super().__init__(**kwargs)

    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (16, 12))

        self.ax_mesh = self.fig.add_axes([.1, .34, .86, .62])
        self.axman_wavefunction.assign_axis(self.ax_mesh)

        self.ax_lower = self.fig.add_axes([.065, .065, .87, .2])
        self.axman_lower.assign_axis(self.ax_lower)

        self.axis_managers += [self.axman_wavefunction, self.axman_lower]

        super()._initialize_figure()


# class PhiSliceMeshAxis(QuantumMeshAxis):
#     def initialize_axis(self):
#         unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)
#
#         self.axis.set_theta_zero_location('N')
#         self.axis.set_theta_direction('clockwise')
#         self.axis.set_rlabel_position(80)
#
#         self.axis.grid(True, color = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], **COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
#         # self.axis.grid(True, color = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], linestyle = ':', linewidth = 2, alpha = 0.8)  # change grid color to make it show up against the colormesh
#         angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
#         self.axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)
#
#         self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
#         self.axis.tick_params(axis = 'y', which = 'major', colors = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], pad = 3)  # make r ticks a color that shows up against the colormesh
#
#         self.axis.set_rlabel_position(80)
#
#         max_yticks = 5
#         yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
#         self.axis.yaxis.set_major_locator(yloc)
#
#         plt.gcf().canvas.draw()  # must draw early to modify the axis text
#
#         tick_labels = self.axis.get_yticklabels()
#         for t in tick_labels:
#             t.set_text(t.get_text() + r'${}$'.format(unit_name))
#             self.axis.set_yticklabels(tick_labels)
#
#         self.axis.set_rmax((self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / unit_value)
#
#         self.axis.axis('tight')
#
#         super().initialize_axis()
#
#         self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(), *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)


# class SphericalSlicePhiSliceMeshAxis(PhiSliceMeshAxis):
#     def initialize(self):
#         self.mesh, self.mesh_mirror = self.sim.mesh.attach_g2_to_axis(self.axis, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit,
#                                                                       distance_unit = self.distance_unit,
#                                                                       animated = True)
#
#         self.redraw += [self.mesh, self.mesh_mirror]
#
#         super(SphericalSlicePhiSliceMeshAxis, self).initialize()
#
#     def update(self):
#         self.sim.mesh.update_g2_mesh(self.mesh, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)
#         self.sim.mesh.update_g2_mesh(self.mesh_mirror, normalize = self.renormalize, log = self.log_g, plot_limit = self.plot_limit)
#
#         super(SphericalSlicePhiSliceMeshAxis, self).update()


# class SphericalHarmonicPhiSliceMeshAxis(PhiSliceMeshAxis):
class SphericalHarmonicPhiSliceMeshAxis(QuantumMeshAxis):
    def __init__(self,
                 slicer = 'get_mesh_slicer_spatial',
                 **kwargs):
        super().__init__(slicer = slicer, **kwargs)

    def initialize_axis(self):
        if self.which == 'g':
            self.norm.equator_magnitude = np.max(np.abs(self.sim.mesh.g) / core.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)

        self.mesh = self.attach_method(self.axis,
                                       colormap = self.colormap,
                                       norm = self.norm,
                                       shading = self.shading,
                                       plot_limit = self.plot_limit,
                                       distance_unit = self.distance_unit,
                                       slicer = self.slicer,
                                       animated = True)
        self.redraw += [self.mesh]

        unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)

        self.axis.set_theta_zero_location('N')
        self.axis.set_theta_direction('clockwise')
        self.axis.set_rlabel_position(80)

        self.axis.grid(True, color = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], **COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        self.axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        self.axis.tick_params(axis = 'y', which = 'major', colors = si.plots.CMAP_TO_OPPOSITE[self.colormap.name], pad = 3)  # make r ticks a color that shows up against the colormesh

        self.axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        self.axis.yaxis.set_major_locator(yloc)

        plt.gcf().canvas.draw()  # must draw early to modify the axis text

        tick_labels = self.axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + r'${}$'.format(unit_name))
            self.axis.set_yticklabels(tick_labels)

        self.axis.set_rmax((self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / unit_value)

        self.axis.axis('tight')

        super().initialize_axis()

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(), *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)


# class AngularMomentumDecompositionAxis(si.AxisManager):
#     def __init__(self, *args, renormalize_l_decomposition = True, **kwargs):
#         self.renormalize_l_decomposition = renormalize_l_decomposition
#
#         super(AngularMomentumDecompositionAxis, self).__init__(*args, **kwargs)
#
#     def initialize(self):
#         l_plot = self.sim.mesh.norm_by_l
#         if self.renormalize_l_decomposition:
#             l_plot /= self.sim.mesh.norm()
#         self.ang_mom_bar = self.axis.bar(self.sim.mesh.l, l_plot,
#                                          align = 'center', color = '.5',
#                                          animated = True)
#
#         self.redraw += [*self.ang_mom_bar]
#
#         self.axis.yaxis.grid(True, zorder = 10)
#
#         self.axis.set_xlabel(r'Orbital Angular Momentum $\ell$', fontsize = 22)
#         l_label = r'$\left| \left\langle \psi | Y^{\ell}_0 \right\rangle \right|^2$'
#         if self.renormalize_l_decomposition:
#             l_label += r'$/\left\langle\psi|\psi\right\rangle$'
#         self.axis.set_ylabel(l_label, fontsize = 22)
#         self.axis.yaxis.set_label_position('right')
#
#         self.axis.set_ylim(0, 1)
#         self.axis.set_xlim(np.min(self.sim.mesh.l) - 0.4, np.max(self.sim.mesh.l) + 0.4)
#
#         self.axis.tick_params(labelleft = False, labelright = True)
#         self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)
#
#         self.redraw += [*self.axis.yaxis.get_gridlines()]
#
#         super(AngularMomentumDecompositionAxis, self).initialize()
#
#     def update(self):
#         l_plot = self.sim.mesh.norm_by_l
#         if self.renormalize_l_decomposition:
#             l_plot /= self.sim.norm_vs_time[self.sim.data_time_index]
#         for bar, height in zip(self.ang_mom_bar, l_plot):
#             bar.set_height(height)
#
#         super(AngularMomentumDecompositionAxis, self).update()
#
#
# class ColorBarAxis(si.AxisManager):
#     def __init__(self, *args, colorable, **kwargs):
#         self.colorable = colorable
#
#         super(ColorBarAxis, self).__init__(*args, **kwargs)
#
#     def initialize(self):
#         self.cbar = plt.colorbar(mappable = self.colorable, cax = self.axis)
#         self.cbar.ax.tick_params(labelsize = 14)
#
#         super(ColorBarAxis, self).initialize()


class PhiSliceAnimator(WavefunctionSimulationAnimator):
    def __init__(self,
                 axman_lower_right = ElectricPotentialAxis(),
                 axman_upper_right = None,
                 **kwargs):
        self.axman_lower_right = axman_lower_right
        self.axman_upper_right = axman_upper_right

        super().__init__(**kwargs)

    def _initialize_figure(self):
        self.fig = plt.figure(figsize = (20, 12))

        self.ax_mesh = self.fig.add_axes([.05, .05, (12 / 20) - 0.05, .9], projection = 'polar')
        self.axman_wavefunction.assign_axis(self.ax_mesh)
        self.axis_managers.append(self.axman_wavefunction)

        if self.axman_lower_right is not None:
            lower_legend_kwargs = dict(bbox_to_anchor = (1., 1.25),
                                       loc = 'lower right',
                                       borderaxespad = 0.0,
                                       fontsize = 20,
                                       fancybox = True,
                                       framealpha = .1)
            self.axman_lower_right.legend_kwargs.update(lower_legend_kwargs)
            self.ax_lower_right = self.fig.add_axes([.575, .075, .36, .15])
            self.axman_lower_right.assign_axis(self.ax_lower_right)
            self.axis_managers.append(self.axman_lower_right)

        if self.axman_upper_right is not None:
            upper_legend_kwargs = dict(bbox_to_anchor = (1., -.25),
                                       loc = 'upper right',
                                       borderaxespad = 0.0,
                                       fontsize = 20,
                                       fancybox = True,
                                       framealpha = .1)
            self.axman_upper_right.legend_kwargs.update(upper_legend_kwargs)
            self.ax_upper_right = self.fig.add_axes([.575, .8, .36, .15])
            self.axman_upper_right.assign_axis(self.ax_upper_right)
            self.axis_managers.append(self.axman_upper_right)

        # self.ax_mesh.initialize()  # must pre-initialize so that the colobar can see the colormesh
        # self.ax_cbar = ColorBarAxis(self.fig.add_axes([.65, .25, .03, .5]), self.sim, colorable = self.ax_mesh.mesh)

        # self.axis_managers += [self.ax_mesh, self.ax_metrics, self.ax_cbar]

        plt.figtext(.075, .9, r'$|g|^2$', fontsize = 50)

        # plt.figtext(.8, .6, r'Initial State: ${}$'.format(self.spec.initial_state.tex_str), fontsize = 22)

        self.time_text = plt.figtext(.8, .49, r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)), fontsize = 30, animated = True)
        self.redraw += [self.time_text]

        super()._initialize_figure()

    def _update_data(self):
        self.time_text.set_text(r'$t = {}$ as'.format(uround(self.sim.time, asec, 3)))

        super()._update_data()

# # class SphericalSliceAnimator(PhiSliceAnimator):
# #     mesh_axis_type = SphericalSlicePhiSliceMeshAxis
#
#
# class SphericalHarmonicAnimator(PhiSliceAnimator):
#     pass
#     # def __init__(self, top_right_axis_manager_type = TestStateStackplot, top_right_axis_kwargs = None, **kwargs):
#     #     self.top_right_axis_manager_type = top_right_axis_manager_type
#     #     if top_right_axis_kwargs is None:
#     #         top_right_axis_kwargs = {}
#     #     self.top_right_axis_kwargs = top_right_axis_kwargs
#     #
#     #     super(SphericalHarmonicAnimator, self).__init__(**kwargs)
#
#     # def _initialize_figure(self):
#     #     super(SphericalHarmonicAnimator, self)._initialize_figure()
#
#     # self.top_right_axis = self.top_right_axis_manager_type(self.fig.add_axes([.56, .84, .39, .11]), self.sim, **self.top_right_axis_kwargs)
#
#     # self.axis_managers += [self.top_right_axis]
