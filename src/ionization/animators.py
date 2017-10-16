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
    **si.vis.COLORMESH_GRID_KWARGS,
    **dict(
        linestyle = ':',
        linewidth = 1.5,
        alpha = 0.6
    )
}


class ElectricPotentialPlotAxis(si.vis.AxisManager):
    def __init__(self,
                 time_unit = 'asec',
                 show_electric_field = True,
                 electric_field_unit = 'AEF',
                 show_vector_potential = False,
                 vector_potential_unit = 'atomic_momentum',
                 linewidth = 3,
                 show_y_label = False,
                 show_ticks_bottom = True,
                 show_ticks_top = False,
                 show_ticks_right = True,
                 show_ticks_left = True,
                 grid_kwargs = None,
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

        self.linewidth = linewidth

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc = 'lower left',
            fontsize = 30,
            fancybox = True,
            framealpha = 0,
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {**si.vis.GRID_KWARGS, **grid_kwargs}

        super().__init__()

    def initialize_axis(self):
        self.time_line = self.axis.axvline(
            x = self.sim.time / self.time_unit_value,
            color = 'gray',
            animated = True
        )
        self.redraw.append(self.time_line)

        if self.show_electric_field:
            self.electric_field_line, = self.axis.plot(
                self.sim.data_times / self.time_unit_value,
                self.sim.electric_field_amplitude_vs_time / self.electric_field_unit_value,
                label = fr'${core.LATEX_EFIELD}(t)$',
                color = core.COLOR_ELECTRIC_FIELD, linewidth = self.linewidth,
                animated = True,
            )

            self.redraw.append(self.electric_field_line)

        if self.show_vector_potential:
            self.vector_potential_line, = self.axis.plot(
                self.sim.data_times / self.time_unit_value,
                proton_charge * self.sim.vector_potential_amplitude_vs_time / self.vector_potential_unit_value,
                label = fr'$q \, {core.LATEX_AFIELD}(t)$',
                color = core.COLOR_VECTOR_POTENTIAL, linewidth = self.linewidth, linestyle = '--',
                animated = True,
            )

            self.redraw.append(self.vector_potential_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(fr'Time $t$ (${self.time_unit_latex}$)', fontsize = 26)

        if self.show_y_label:
            self.axis.set_ylabel('Wavefunction Metric', fontsize = 26)

        self.axis.tick_params(
            labelbottom = self.show_ticks_bottom,
            labeltop = self.show_ticks_top,
            labelright = self.show_ticks_right,
            labelleft = self.show_ticks_left
        )

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit_value, self.sim.data_times[-1] / self.time_unit_value)

        data = []
        if self.show_electric_field:
            data.append(self.spec.electric_potential.get_electric_field_amplitude(self.sim.times) / self.electric_field_unit_value)
        if self.show_vector_potential:
            data.append(proton_charge * self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times) / self.vector_potential_unit_value)

        y_lower_limit, y_upper_limit = si.vis.set_axis_limits(self.axis,
                                                              *data,
                                                              pad = 0.05,
                                                              direction = 'y',
                                                              )

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 16)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def update_axis(self):
        if self.show_electric_field:
            self.electric_field_line.set_ydata(self.sim.electric_field_amplitude_vs_time / self.electric_field_unit_value)
        if self.show_vector_potential:
            self.vector_potential_line.set_ydata(proton_charge * self.sim.vector_potential_amplitude_vs_time / self.vector_potential_unit_value)

        self.time_line.set_xdata(self.sim.time / self.time_unit_value)

        super().update_axis()

    def info(self):
        info = super().info()

        info.add_field('Show Electric Field', self.show_electric_field)
        if self.show_electric_field:
            info.add_field('Electric Field Unit', self.electric_field_unit)
        info.add_field('Show Vector Potential', self.show_vector_potential)
        if self.show_vector_potential:
            info.add_field('Vector Potential Unit', self.vector_potential_unit)
        info.add_field('Time Unit', self.time_unit)

        return info


class StackplotAxis(si.vis.AxisManager):
    def __init__(self,
                 show_norm = True,
                 time_unit = 'asec',
                 y_label = None,
                 show_ticks_bottom = True,
                 show_ticks_top = False,
                 show_ticks_right = True,
                 show_ticks_left = True,
                 grid_kwargs = None,
                 legend_kwargs = None):
        self.show_norm = show_norm

        self.time_unit = time_unit
        self.time_unit_value, self.time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

        self.y_label = y_label
        self.show_ticks_bottom = show_ticks_bottom
        self.show_ticks_top = show_ticks_top
        self.show_ticks_right = show_ticks_right
        self.show_ticks_left = show_ticks_left

        if legend_kwargs is None:
            legend_kwargs = {}
        legend_defaults = dict(
            loc = 'lower left',
            fontsize = 30,
            fancybox = True,
            framealpha = 0,
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {**si.vis.GRID_KWARGS, **grid_kwargs}

        super().__init__()

    def initialize_axis(self):
        if self.show_norm:
            self.norm_line, = self.axis.plot(
                self.sim.data_times / self.time_unit_value,
                self.sim.norm_vs_time,
                label = r'$\left\langle \Psi|\psi \right\rangle$',
                color = 'black',
                linewidth = 3
            )

            self.redraw.append(self.norm_line)

        self._initialize_stackplot()

        self.time_line = self.axis.axvline(x = self.sim.time / self.time_unit_value,
                                           color = 'gray',
                                           linewidth = 1,
                                           animated = True)
        self.redraw.append(self.time_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(fr'Time $t$ (${self.time_unit_latex}$)', fontsize = 26)

        if self.y_label is not None:
            self.axis.set_ylabel(self.y_label, fontsize = 26)

        self.axis.tick_params(labelbottom = self.show_ticks_bottom, labeltop = self.show_ticks_top, labelright = self.show_ticks_right, labelleft = self.show_ticks_left)

        self.axis.set_xlim(self.sim.data_times[0] / self.time_unit_value, self.sim.data_times[-1] / self.time_unit_value)
        self.axis.set_ylim(0, 1.05)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 16)

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def _get_stackplot_data_and_labels(self):
        raise NotImplementedError

    def _initialize_stackplot(self):
        data, labels = self._get_stackplot_data_and_labels()

        self.overlaps_stackplot = self.axis.stackplot(self.sim.data_times / self.time_unit_value,
                                                      *data,
                                                      labels = labels,
                                                      animated = True)

        self.redraw += [*self.overlaps_stackplot]

    def _update_stackplot_lines(self):
        for x in self.overlaps_stackplot:
            self.redraw.remove(x)
            x.remove()

        self.axis.set_prop_cycle(None)

        data, labels = self._get_stackplot_data_and_labels()

        self.overlaps_stackplot = self.axis.stackplot(
            self.sim.data_times / self.time_unit_value,
            *data,
            labels = labels,
            animated = True
        )

        self.redraw = [*self.overlaps_stackplot] + self.redraw

    def update_axis(self):
        if self.show_norm:
            self.norm_line.set_ydata(self.sim.norm_vs_time)

        self._update_stackplot_lines()

        self.time_line.set_xdata(self.sim.time / self.time_unit_value)

        super().update_axis()

    def info(self):
        info = super().info()

        info.add_field('Show Norm', self.show_norm)
        info.add_field('Time Unit', self.time_unit)

        return info


class TestStateStackplotAxis(StackplotAxis):
    def __init__(self,
                 states = None,
                 **kwargs):
        self.states = states
        if not callable(self.states):
            self.states = tuple(sorted(states))
        if len(self.states) > 8:
            logger.warning(f'Using more than 8 states in a {self.__class__.__name__} is ill-advised')

        super().__init__(**kwargs)

    def _get_stackplot_data_and_labels(self):
        state_overlaps = self.sim.state_overlaps_vs_time
        if self.states is not None:
            if callable(self.states):
                data = (overlap for state, overlap in sorted(state_overlaps.items()) if self.states(state))
            else:
                states = set(self.states)
                data = (overlap for state, overlap in sorted(state_overlaps.items()) if state in states or (state.numeric and state.analytic_state in states))
        else:
            data = (overlap for state, overlap in sorted(state_overlaps.items()))

        labels = (r'$\left| \left\langle \Psi| {} \right\rangle \right|^2$'.format(state.latex) for state in sorted(state_overlaps))

        return data, labels


class WavefunctionStackplotAxis(StackplotAxis):
    def __init__(self,
                 states = None,
                 **kwargs):
        if states is None:
            states = ()
        self.states = sorted(states)

        super().__init__(**kwargs)

    def _get_stackplot_data_and_labels(self):
        state_overlaps = self.sim.state_overlaps_vs_time

        selected_state_overlaps = {state: overlap for state, overlap in sorted(state_overlaps.items()) if state in self.states or (state.numeric and state.analytic_state in self.states)}
        overlap_len = len(list(state_overlaps.values())[0])  # ugly, but I don't see a way around it

        data = [
            *(overlap for state, overlap in sorted(selected_state_overlaps.items())),
            sum((overlap for state, overlap in state_overlaps.items() if state.bound and (state not in self.states and (not state.numeric or state.analytic_state not in self.states))),
                np.zeros(overlap_len)),
            sum((overlap for state, overlap in state_overlaps.items() if state.free and (state not in self.states and (not state.numeric or state.analytic_state not in self.states))),
                np.zeros(overlap_len)),
        ]

        labels = (
            *(r'$ \left| \left\langle \Psi | {} \right\rangle \right|^2 $'.format(state.latex) for state, overlap in sorted(selected_state_overlaps.items())),
            r'$ \sum_{\mathrm{other \, bound}} \; \left| \left\langle \Psi | \psi_{{n, \, \ell}} \right\rangle \right|^2 $',
            fr'$ \sum_{{ \mathrm{{other \, free}} }} \; \left| \left\langle \Psi | \phi_{{E, \, \ell}} \right\rangle \right|^2 $',
        )

        return data, labels


class AngularMomentumDecompositionAxis(si.vis.AxisManager):
    def __init__(self, renormalize_l_decomposition = False, maximum_l = None):
        self.renormalize_l_decomposition = renormalize_l_decomposition
        self.maximum_l = maximum_l
        self.slice = slice(self.maximum_l + 1)

        super().__init__()

    def initialize_axis(self):
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.mesh.norm()
        self.ang_mom_bar = self.axis.bar(
            self.sim.mesh.l[self.slice],
            l_plot[self.slice],
            align = 'center',
            color = '.5',
            animated = True
        )

        self.redraw += [*self.ang_mom_bar]

        self.axis.yaxis.grid(True, **si.vis.GRID_KWARGS)

        self.axis.set_xlabel(r'Orbital Angular Momentum $\ell$', fontsize = 22)
        l_label = r'$\left| \left\langle \Psi | Y^{\ell}_0 \right\rangle \right|^2$'
        if self.renormalize_l_decomposition:
            l_label += r'$/\left\langle\Psi|\Psi\right\rangle$'
        self.axis.set_ylabel(l_label, fontsize = 22)
        self.axis.yaxis.set_label_position('right')

        self.axis.set_ylim(0, 1)
        self.axis.set_xlim(np.min(self.sim.mesh.l[self.slice]) - 0.4, np.max(self.sim.mesh.l[self.slice]) + 0.4)

        self.axis.xaxis.set_major_locator(plt.MaxNLocator(integer = True))

        self.axis.tick_params(labelleft = False, labelright = True)
        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)

        self.redraw += [*self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def update_axis(self):
        l_plot = self.sim.mesh.norm_by_l[self.slice]
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.norm_vs_time[self.sim.data_time_index]
        for bar, height in zip(self.ang_mom_bar, l_plot):
            bar.set_height(height)

        super().update_axis()


class ColorBarAxis(si.vis.AxisManager):
    def assign_colorable(self,
                         colorable,
                         fontsize = 14):
        self.colorable = colorable
        self.fontsize = fontsize

    def initialize_axis(self):
        self.cbar = plt.colorbar(mappable = self.colorable, cax = self.axis)
        self.cbar.ax.tick_params(labelsize = self.fontsize)

        super().initialize_axis()


class QuantumMeshAxis(si.vis.AxisManager):
    def __init__(self,
                 which = 'g2',
                 colormap = core.COLORMAP_WAVEFUNCTION,
                 norm = si.vis.AbsoluteRenormalize(),
                 plot_limit = None,
                 distance_unit = 'bohr_radius',
                 shading = 'gouraud',
                 slicer = 'get_mesh_slicer',
                 grid_kwargs = None):
        self.which = which
        self.colormap = colormap
        self.norm = norm
        self.plot_limit = plot_limit
        self.distance_unit = distance_unit
        self.shading = shading
        self.slicer = slicer

        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {**COLORMESH_GRID_KWARGS, 'color': si.vis.CMAP_TO_OPPOSITE[self.colormap.name], **grid_kwargs}

        super().__init__()

    def initialize(self, simulation):
        self.attach_method = getattr(simulation.mesh, f'attach_{self.which}_to_axis')
        self.update_method = getattr(simulation.mesh, f'update_{self.which}_mesh')

        super().initialize(simulation)

    def update_axis(self):
        self.update_method(
            self.mesh,
            shading = self.shading,
            plot_limit = self.plot_limit,
            slicer = self.slicer,
            norm = self.norm
        )

        super().update_axis()

    def info(self):
        info = super().info()

        info.add_field('Plotting', self.which)
        info.add_field('Colormap', self.colormap.name)
        info.add_field('Normalization', self.norm.__class__.__name__)
        info.add_field('Plot Limit', f'{uround(self.plot_limit, bohr_radius)} Bohr radii | {uround(self.plot_limit, nm)} nm' if self.plot_limit is not None else 'none')
        info.add_field('Distance Unit', self.distance_unit)
        info.add_field('Shading', self.shading)

        return info


class LineMeshAxis(QuantumMeshAxis):
    def __init__(self,
                 which = 'psi2',
                 # show_potential = False,
                 log = False,
                 **kwargs):
        # self.show_potential = show_potential

        super().__init__(which = which, **kwargs)
        self.log = log

    def initialize_axis(self):
        unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)

        self.mesh = self.attach_method(
            self.axis,
            colormap = self.colormap,
            norm = self.norm,
            shading = self.shading,
            plot_limit = self.plot_limit,
            distance_unit = self.distance_unit,
            slicer = self.slicer,
            animated = True,
            linewidth = 3,
        )
        self.redraw.append(self.mesh)

        if self.log:
            self.axis.set_yscale('log')
            self.axis.set_ylim(bottom = 1e-15)

        # TODO: code for show_potential

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(r'$x$ (${}$)'.format(unit_name), fontsize = 24)
        plot_labels = {
            'g2': r'$ \left| g \right|^2 $',
            'psi2': r'$ \left| \Psi \right|^2 $',
            'g': r'$ g $',
            'psi': r'$ \Psi $',
            'fft': r'$ \phi $',
        }
        self.axis.set_ylabel(plot_labels[self.which], fontsize = 30)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)
        self.axis.tick_params(labelright = True, labeltop = True)

        slice = getattr(self.sim.mesh, self.slicer)(self.plot_limit)
        x = self.sim.mesh.x_mesh[slice]
        x_lower_limit, x_upper_limit = np.nanmin(x), np.nanmax(x)
        self.axis.set_xlim(x_lower_limit / unit_value, x_upper_limit / unit_value)


        # self.axis.axis('tight')

        self.redraw += [*self.axis.xaxis.get_gridlines(),
                        *self.axis.yaxis.get_gridlines()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        super().initialize_axis()

    def update_axis(self):
        # TODO: code for show_potential
        super().update_axis()


class CylindricalSliceMeshAxis(QuantumMeshAxis):
    def initialize_axis(self):
        unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)

        if self.which == 'g':
            self.norm.equator_magnitude = np.max(np.abs(self.sim.mesh.g) / core.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)

        self.mesh = self.attach_method(
            self.axis,
            colormap = self.colormap,
            norm = self.norm,
            shading = self.shading,
            plot_limit = self.plot_limit,
            distance_unit = self.distance_unit,
            slicer = self.slicer,
            animated = True,
        )
        self.redraw.append(self.mesh)

        self.axis.grid(True, **self.grid_kwargs)  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r'$z$ (${}$)'.format(unit_name), fontsize = 24)
        self.axis.set_ylabel(r'$\rho$ (${}$)'.format(unit_name), fontsize = 24)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)

        self.axis.axis('tight')

        super().initialize_axis()

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(),
                        *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        if self.which not in ('g', 'psi'):
            divider = make_axes_locatable(self.axis)
            cax = divider.append_axes("right", size = "2%", pad = 0.05)
            self.cbar = plt.colorbar(cax = cax, mappable = self.mesh)
            self.cbar.ax.tick_params(labelsize = 20)
        else:
            logger.warning('show_colorbar cannot be used with nonlinear colormaps')


class SphericalHarmonicPhiSliceMeshAxis(QuantumMeshAxis):
    def __init__(self,
                 slicer = 'get_mesh_slicer_spatial',
                 **kwargs):
        self.tick_labels = None

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
        self.redraw.append(self.mesh)

        unit_value, unit_name = get_unit_value_and_latex_from_unit(self.distance_unit)

        self.axis.set_theta_zero_location('N')
        self.axis.set_theta_direction('clockwise')
        self.axis.set_rlabel_position(80)

        self.axis.grid(True, **self.grid_kwargs)  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        self.axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        self.axis.tick_params(axis = 'both', which = 'major', labelsize = 20)  # increase size of tick labels
        self.axis.tick_params(axis = 'y', which = 'major', colors = si.vis.CMAP_TO_OPPOSITE[self.colormap.name], pad = 3)  # make r ticks a color that shows up against the colormesh

        self.axis.set_rlabel_position(80)

        if self.tick_labels is None:
            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            self.axis.yaxis.set_major_locator(yloc)

            plt.gcf().canvas.draw()  # must draw early to modify the axis text

            self.tick_labels = self.axis.get_yticklabels()
            for t in self.tick_labels:
                t.set_text(t.get_text() + r'${}$'.format(unit_name))
            self.axis.set_yticklabels(self.tick_labels)

        self.axis.set_rmax((self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / unit_value)

        self.axis.axis('tight')

        super().initialize_axis()

        self.redraw += [*self.axis.xaxis.get_gridlines(), *self.axis.yaxis.get_gridlines(),
                        *self.axis.yaxis.get_ticklabels()]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)


class WavefunctionSimulationAnimator(si.vis.Animator):
    def __init__(self, *,
                 axman_wavefunction,
                 **kwargs):
        super().__init__(**kwargs)

        self.axman_wavefunction = axman_wavefunction
        self.axis_managers.append(self.axman_wavefunction)

    def __str__(self):
        return si.utils.field_str(self, 'postfix', 'axis_managers')

    def __repr__(self):
        return self.__str__()


class RectangleAnimator(WavefunctionSimulationAnimator):
    def __init__(self,
                 axman_lower = ElectricPotentialPlotAxis(),
                 fig_dpi_scale = 1,
                 **kwargs):
        super().__init__(**kwargs)

        self.axman_lower = axman_lower
        self.axis_managers.append(self.axman_lower)

        self.fig_dpi_scale = fig_dpi_scale

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(fig_width = 16, fig_height = 12, fig_dpi_scale = self.fig_dpi_scale)

        self.ax_mesh = self.fig.add_axes([.1, .34, .84, .6])
        self.axman_wavefunction.assign_axis(self.ax_mesh)

        self.ax_lower = self.fig.add_axes([.065, .065, .87, .2])
        self.axman_lower.assign_axis(self.ax_lower)

        super()._initialize_figure()


class RectangleSplitLowerAnimator(WavefunctionSimulationAnimator):
    def __init__(self,
                 axman_lower_left = ElectricPotentialPlotAxis(),
                 axman_lower_right = WavefunctionStackplotAxis(),
                 fig_dpi_scale = 1,
                 **kwargs):
        super().__init__(**kwargs)

        self.axman_lower_left = axman_lower_left
        self.axman_lower_right = axman_lower_right

        self.axis_managers += [self.axman_lower_left, self.axman_lower_right]

        self.fig_dpi_scale = fig_dpi_scale

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(fig_width = 16, fig_height = 12, fig_dpi_scale = self.fig_dpi_scale)

        self.ax_mesh = self.fig.add_axes([.1, .34, .84, .6])
        self.axman_wavefunction.assign_axis(self.ax_mesh)

        self.ax_lower_left = self.fig.add_axes([.06, .06, .4, .2])
        self.axman_lower_left.assign_axis(self.ax_lower_left)

        self.ax_lower_right = self.fig.add_axes([.56, .06, .4, .2])
        self.axman_lower_right.assign_axis(self.ax_lower_right)

        super()._initialize_figure()


class PolarAnimator(WavefunctionSimulationAnimator):
    def __init__(self,
                 axman_lower_right = ElectricPotentialPlotAxis(),
                 axman_upper_right = WavefunctionStackplotAxis(),
                 axman_colorbar = ColorBarAxis(),
                 fig_dpi_scale = 1,
                 **kwargs):
        super().__init__(**kwargs)

        self.axman_lower_right = axman_lower_right
        self.axman_upper_right = axman_upper_right
        self.axman_colorbar = axman_colorbar

        self.axis_managers += [axman for axman in [self.axman_lower_right, self.axman_upper_right, self.axman_colorbar] if axman is not None]

        self.fig_dpi_scale = fig_dpi_scale

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(fig_width = 20, fig_height = 12, fig_dpi_scale = self.fig_dpi_scale)

        self.ax_wavefunction = self.fig.add_axes([.05, .05, (12 / 20) - 0.05, .9], projection = 'polar')
        self.axman_wavefunction.assign_axis(self.ax_wavefunction)

        if self.axman_lower_right is not None:
            lower_legend_kwargs = dict(bbox_to_anchor = (1., 1.2),
                                       loc = 'lower right',
                                       borderaxespad = 0.0)
            self.axman_lower_right.legend_kwargs.update(lower_legend_kwargs)
            self.ax_lower_right = self.fig.add_axes([.575, .075, .36, .15])
            self.axman_lower_right.assign_axis(self.ax_lower_right)

        if self.axman_upper_right is not None:
            upper_legend_kwargs = dict(bbox_to_anchor = (1., -.35),
                                       loc = 'upper right',
                                       borderaxespad = 0.0)
            try:
                self.axman_upper_right.legend_kwargs.update(upper_legend_kwargs)
            except AttributeError:
                pass
            self.ax_upper_right = self.fig.add_axes([.575, .8, .36, .15])
            self.axman_upper_right.assign_axis(self.ax_upper_right)

        if self.axman_colorbar is not None:
            if self.axman_wavefunction.which not in ('g', 'psi'):
                self.axman_wavefunction.initialize(self.sim)  # must pre-initialize so that the colorbar can see the colormesh
                self.axman_colorbar.assign_colorable(self.axman_wavefunction.mesh)
                self.ax_colobar = self.fig.add_axes([.65, .35, .02, .35])
                self.axman_colorbar.assign_axis(self.ax_colobar)
            else:
                logger.warning('ColorbarAxis cannot be used with nonlinear colormaps')

        plot_labels = {
            'g2': r'$ \left| g \right|^2 $',
            'psi2': r'$ \left| \Psi \right|^2 $',
            'g': r'$ g $',
            'psi': r'$ \Psi $',
        }
        plt.figtext(.075, .9, plot_labels[self.axman_wavefunction.which], fontsize = 50)

        self.time_text = plt.figtext(.6, .3,
                                     r'$t = {} \, \mathrm{{as}}$'.format(uround(self.sim.time, asec, 1)),
                                     fontsize = 30,
                                     animated = True)
        self.redraw.append(self.time_text)

        super()._initialize_figure()

    def _update_data(self):
        self.time_text.set_text(r'$t = {} \, \mathrm{{as}}$'.format(uround(self.sim.time, asec, 1)))

        super()._update_data()
