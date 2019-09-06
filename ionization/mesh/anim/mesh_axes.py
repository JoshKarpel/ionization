import logging

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import simulacra as si
import simulacra.units as u

from ... import vis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COLORMESH_GRID_KWARGS = {
    **si.vis.DEFAULT_COLORMESH_GRID_KWARGS,
    **dict(linestyle=":", linewidth=1.5, alpha=0.6),
}


class QuantumMeshAxis(si.vis.AxisManager):
    def __init__(
        self,
        which="g",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        plot_limit=None,
        distance_unit="bohr_radius",
        shading="flat",
        slicer="get_mesh_slicer",
        show_grid=True,
        grid_kwargs=None,
        axis_off=False,
    ):
        self.which = which
        self.colormap = colormap
        self.norm = norm
        self.plot_limit = plot_limit
        self.distance_unit = distance_unit
        self.shading = shading
        self.slicer = slicer

        self.show_grid = show_grid
        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {
            **COLORMESH_GRID_KWARGS,
            "color": si.vis.CMAP_TO_OPPOSITE[self.colormap.name],
            **grid_kwargs,
        }
        self.axis_off = axis_off

        super().__init__()

    def initialize(self, simulation):
        self.attach_method = getattr(
            simulation.mesh.plot, f"attach_{self.which}_to_axis"
        )
        self.update_method = getattr(simulation.mesh.plot, f"update_{self.which}_mesh")

        super().initialize(simulation)

    def update_axis(self):
        self.update_method(
            self.mesh,
            shading=self.shading,
            plot_limit=self.plot_limit,
            slicer=self.slicer,
            norm=self.norm,
        )

        super().update_axis()

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Plotting", self.which)
        info.add_field("Colormap", self.colormap.name)
        info.add_field("Normalization", self.norm.__class__.__name__)
        info.add_field(
            "Plot Limit",
            f"{self.plot_limit / u.bohr_radius:.3f} Bohr radii | {self.plot_limit / u.nm:.3f} nm"
            if self.plot_limit is not None
            else "none",
        )
        info.add_field("Distance Unit", self.distance_unit)
        info.add_field("Shading", self.shading)

        return info


class LineMeshAxis(QuantumMeshAxis):
    def __init__(
        self,
        which="psi2",
        # show_potential = False,
        log=False,
        **kwargs,
    ):
        super().__init__(which=which, **kwargs)
        self.log = log
        # self.show_potential = show_potential

    def initialize_axis(self):
        unit_value, unit_name = u.get_unit_value_and_latex(self.distance_unit)

        self.mesh = self.attach_method(
            self.axis,
            colormap=self.colormap,
            norm=self.norm,
            shading=self.shading,
            plot_limit=self.plot_limit,
            distance_unit=self.distance_unit,
            slicer=self.slicer,
            animated=True,
            linewidth=3,
        )
        self.redraw.append(self.mesh)

        if self.log:
            self.axis.set_yscale("log")
            self.axis.set_ylim(bottom=1e-15)

        # TODO: code for show_potential

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(r"$ z $ ($ {} $)".format(unit_name), fontsize=24)
        plot_labels = {
            "g2": r"$ \left| g \right|^2 $",
            "psi2": r"$ \left| \Psi \right|^2 $",
            "g": r"$ g $",
            "psi": r"$ \Psi $",
            "fft": r"$ \phi $",
        }
        self.axis.set_ylabel(plot_labels[self.which], fontsize=30)

        self.axis.tick_params(axis="both", which="major", labelsize=20)
        self.axis.tick_params(labelright=True, labeltop=True)

        slice = getattr(self.sim.mesh, self.slicer)(self.plot_limit)
        z = self.sim.mesh.z_mesh[slice]
        z_lower_limit, z_upper_limit = np.nanmin(z), np.nanmax(z)
        self.axis.set_xlim(z_lower_limit / unit_value, z_upper_limit / unit_value)

        self.redraw += [
            *self.axis.xaxis.get_gridlines(),
            *self.axis.yaxis.get_gridlines(),
        ]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        super().initialize_axis()

    def update_axis(self):
        # TODO: code for show_potential
        super().update_axis()


class RectangleMeshAxis(QuantumMeshAxis):
    def initialize_axis(self):
        unit_value, unit_name = u.get_unit_value_and_latex(self.distance_unit)

        if self.which == "g":
            self.norm.equator_magnitude = np.max(
                np.abs(self.sim.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
            )

        self.mesh = self.attach_method(
            self.axis,
            colormap=self.colormap,
            norm=self.norm,
            shading=self.shading,
            plot_limit=self.plot_limit,
            distance_unit=self.distance_unit,
            slicer=self.slicer,
            animated=True,
        )

        pot = self.spec.internal_potential(
            x=self.sim.mesh.x_mesh, z=self.sim.mesh.z_mesh, r=self.sim.mesh.r_mesh
        )
        pot[np.abs(pot) < 0.1 * np.nanmax(np.abs(pot))] = np.NaN
        self.potential_mesh = self.sim.mesh.plot.attach_mesh_to_axis(
            self.axis,
            np.ma.masked_invalid(pot),
            colormap=plt.get_cmap("coolwarm_r"),
            norm=plt.Normalize(vmin=-np.max(np.abs(pot)), vmax=np.max(np.abs(pot))),
            shading=self.shading,
            plot_limit=self.plot_limit,
            distance_unit=self.distance_unit,
            slicer=self.slicer,
            animated=True,
        )
        self.redraw.append(self.mesh)
        self.redraw.append(self.potential_mesh)

        self.axis.set_xlabel(r"$x$ (${}$)".format(unit_name), fontsize=24)
        self.axis.set_ylabel(r"$z$ (${}$)".format(unit_name), fontsize=24)

        self.axis.tick_params(axis="both", which="major", labelsize=20)

        self.axis.axis("tight")

        super().initialize_axis()

        if self.which not in ("g", "psi"):
            divider = make_axes_locatable(self.axis)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            self.cbar = plt.colorbar(cax=cax, mappable=self.mesh)
            self.cbar.ax.tick_params(labelsize=20)

        if self.axis_off:
            # self.axis.set_axis_off()
            # self.axis.set_visible(False)
            # self.axis.get_yaxis().set_visible(False)
            self.axis.axis("off")
        else:
            self.redraw += [
                *self.axis.xaxis.get_gridlines(),
                *self.axis.yaxis.get_gridlines(),
            ]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        if self.show_grid:
            self.axis.grid(
                True, **self.grid_kwargs
            )  # change grid color to make it show up against the colormesh


class CylindricalSliceMeshAxis(QuantumMeshAxis):
    def initialize_axis(self):
        unit_value, unit_name = u.get_unit_value_and_latex(self.distance_unit)

        if self.which == "g":
            self.norm.equator_magnitude = np.max(
                np.abs(self.sim.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
            )

        self.mesh = self.attach_method(
            self.axis,
            colormap=self.colormap,
            norm=self.norm,
            shading=self.shading,
            plot_limit=self.plot_limit,
            distance_unit=self.distance_unit,
            slicer=self.slicer,
            animated=True,
        )
        self.redraw.append(self.mesh)

        self.axis.grid(
            True, **self.grid_kwargs
        )  # change grid color to make it show up against the colormesh

        self.axis.set_xlabel(r"$z$ (${}$)".format(unit_name), fontsize=24)
        self.axis.set_ylabel(r"$\rho$ (${}$)".format(unit_name), fontsize=24)

        self.axis.tick_params(axis="both", which="major", labelsize=20)

        self.axis.axis("tight")

        super().initialize_axis()

        self.redraw += [
            *self.axis.xaxis.get_gridlines(),
            *self.axis.yaxis.get_gridlines(),
            *self.axis.yaxis.get_ticklabels(),
        ]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)

        if self.which not in ("g", "psi"):
            divider = make_axes_locatable(self.axis)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            self.cbar = plt.colorbar(cax=cax, mappable=self.mesh)
            self.cbar.ax.tick_params(labelsize=20)
        else:
            logger.warning("show_colorbar cannot be used with nonlinear colormaps")


class SphericalHarmonicPhiSliceMeshAxis(QuantumMeshAxis):
    def __init__(self, slicer="get_mesh_slicer_spatial", **kwargs):
        self.tick_labels = None

        super().__init__(slicer=slicer, **kwargs)

    def initialize_axis(self):
        if self.which == "g":
            self.norm.equator_magnitude = np.max(
                np.abs(self.sim.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
            )

        self.mesh = self.attach_method(
            self.axis,
            colormap=self.colormap,
            norm=self.norm,
            shading=self.shading,
            plot_limit=self.plot_limit,
            distance_unit=self.distance_unit,
            slicer=self.slicer,
            animated=True,
        )
        self.redraw.append(self.mesh)

        unit_value, unit_name = u.get_unit_value_and_latex(self.distance_unit)

        self.axis.set_theta_zero_location("N")
        self.axis.set_theta_direction("clockwise")
        self.axis.set_rlabel_position(80)

        self.axis.grid(
            True, **self.grid_kwargs
        )  # change grid color to make it show up against the colormesh
        angle_labels = [
            "{}\u00b0".format(s)
            for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)
        ]  # \u00b0 is unicode degree symbol
        self.axis.set_thetagrids(np.arange(0, 359, 30), frac=1.075, labels=angle_labels)

        self.axis.tick_params(
            axis="both", which="major", labelsize=20
        )  # increase size of tick labels
        self.axis.tick_params(
            axis="y",
            which="major",
            colors=si.vis.CMAP_TO_OPPOSITE[self.colormap.name],
            pad=3,
        )  # make r ticks a color that shows up against the colormesh

        self.axis.set_rlabel_position(80)

        if self.tick_labels is None:
            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric=False, prune="both")
            self.axis.yaxis.set_major_locator(yloc)

            plt.gcf().canvas.draw()  # must draw early to modify the axis text

            self.tick_labels = self.axis.get_yticklabels()
            for t in self.tick_labels:
                t.set_text(t.get_text() + r"${}$".format(unit_name))
            self.axis.set_yticklabels(self.tick_labels)

        self.axis.set_rmax(
            (self.sim.mesh.r_max - (self.sim.mesh.delta_r / 2)) / unit_value
        )

        self.axis.axis("tight")

        super().initialize_axis()

        self.redraw += [
            *self.axis.xaxis.get_gridlines(),
            *self.axis.yaxis.get_gridlines(),
            *self.axis.yaxis.get_ticklabels(),
        ]  # gridlines must be redrawn over the mesh (it's important that they're AFTER the mesh itself in self.redraw)
