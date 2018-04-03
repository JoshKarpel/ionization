import logging

import matplotlib.pyplot as plt

import simulacra as si
import simulacra.units as u

from . import axes
from ionization import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class WavefunctionSimulationAnimator(si.vis.Animator):
    def __init__(
        self,
        *,
        axman_wavefunction,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.axman_wavefunction = axman_wavefunction
        self.axis_managers.append(self.axman_wavefunction)

    def __str__(self):
        return utils.fmt_fields(self, 'postfix', 'axis_managers')

    def __repr__(self):
        return self.__str__()


class RectangleAnimator(WavefunctionSimulationAnimator):
    def __init__(
        self,
        axman_lower = axes.ElectricPotentialPlotAxis(),
        fig_dpi_scale = 1,
        **kwargs,
    ):
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
    def __init__(
        self,
        axman_lower_left = axes.ElectricPotentialPlotAxis(),
        axman_lower_right = axes.WavefunctionStackplotAxis(),
        fig_dpi_scale = 1,
        **kwargs,
    ):
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


class SquareAnimator(WavefunctionSimulationAnimator):
    def __init__(
        self,
        fig_dpi_scale = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fig_dpi_scale = fig_dpi_scale

    def _initialize_figure(self):
        self.fig = si.vis.get_figure(fig_width = 12, fig_height = 12, fig_dpi_scale = self.fig_dpi_scale)

        self.ax_mesh = self.fig.add_axes([.1, .1, .8, .8])
        self.axman_wavefunction.assign_axis(self.ax_mesh)

        super()._initialize_figure()


class PolarAnimator(WavefunctionSimulationAnimator):
    def __init__(
        self,
        axman_lower_right = axes.ElectricPotentialPlotAxis(),
        axman_upper_right = axes.WavefunctionStackplotAxis(),
        axman_colorbar = axes.ColorBarAxis(),
        fig_dpi_scale = 1,
        time_text_unit: u.Unit = 'asec',
        time_text_digits = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.axman_lower_right = axman_lower_right
        self.axman_upper_right = axman_upper_right
        self.axman_colorbar = axman_colorbar

        self.axis_managers += [axman for axman in [self.axman_lower_right, self.axman_upper_right, self.axman_colorbar] if axman is not None]

        self.fig_dpi_scale = fig_dpi_scale
        self.time_text_unit, self.time_text_unit_tex = u.get_unit_value_and_latex_from_unit(time_text_unit)
        self.time_text_digits = time_text_digits

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

        self.time_text = plt.figtext(
            .6, .3,
            fr'$t = {u.uround(self.sim.time, self.time_text_unit, self.time_text_digits)} \, {self.time_text_unit_tex}$',
            fontsize = 30,
            animated = True
        )
        self.redraw.append(self.time_text)

        super()._initialize_figure()

    def _update_data(self):
        self.time_text.set_text(fr'$t = {u.uround(self.sim.time, self.time_text_unit, self.time_text_digits)} \, {self.time_text_unit_tex}$')

        super()._update_data()
