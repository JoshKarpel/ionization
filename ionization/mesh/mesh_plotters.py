import logging
import collections
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import simulacra as si
import simulacra.units as u

from .. import vis, core
from . import meshes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MeshPlotter:
    def __init__(self, mesh):
        self.mesh = mesh
        self.sim = mesh.sim
        self.spec = mesh.spec

    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: u.Unit = "bohr_radius",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer: str = "get_mesh_slicer",
        **kwargs,
    ):
        raise NotImplementedError

    def attach_g2_to_axis(self, axis: plt.Axes, **kwargs):
        return self.attach_mesh_to_axis(axis, self.mesh.g2, **kwargs)

    def attach_psi2_to_axis(self, axis, **kwargs):
        return self.attach_mesh_to_axis(axis, self.mesh.psi2, **kwargs)

    def attach_g_to_axis(
        self, axis: plt.Axes, colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):
        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        return self.attach_mesh_to_axis(
            axis, self.mesh.g, colormap=colormap, norm=norm, **kwargs
        )

    def attach_psi_to_axis(
        self, axis: plt.Axes, colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):
        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        return self.attach_mesh_to_axis(
            axis, self.mesh.psi, colormap=colormap, norm=norm, **kwargs
        )

    def update_mesh(
        self,
        colormesh,
        updated_mesh,
        plot_limit: Optional[float] = None,
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        slicer: str = "get_mesh_slicer",
        **kwargs,
    ):
        _slice = getattr(self.mesh, slicer)(plot_limit)
        updated_mesh = updated_mesh[_slice]

        try:
            if shading == si.vis.ColormapShader.FLAT:
                updated_mesh = updated_mesh[:-1, :-1]
            colormesh.set_array(updated_mesh.ravel())
        except AttributeError:  # if the mesh is 1D we can't .ravel() it and instead should just set the y data with the mesh
            colormesh.set_ydata(updated_mesh)

    def update_g2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.g2, **kwargs)

    def update_psi2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.psi2, **kwargs)

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.psi, **kwargs)

    def plot_mesh(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: str = "bohr_radius",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer: str = "get_mesh_slicer",
        **kwargs,
    ):
        """kwargs go to figman"""
        raise NotImplementedError

    def g(
        self,
        title: Optional[str] = None,
        name_postfix: str = "",
        colormap=plt.get_cmap("richardson"),
        norm=None,
        **kwargs,
    ):
        if title is None:
            title = r"$g$"
        name = "g" + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        self.plot_mesh(
            self.mesh.g,
            name=name,
            title=title,
            colormap=colormap,
            norm=norm,
            show_colorbar=False,
            **kwargs,
        )

    def psi(
        self, name_postfix="", colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):
        title = r"$\psi$"
        name = "psi" + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.psi) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        self.plot_mesh(
            self.mesh.psi,
            name=name,
            title=title,
            colormap=colormap,
            norm=norm,
            show_colorbar=False,
            **kwargs,
        )

    def g2(self, name_postfix: str = "", title: Optional[str] = None, **kwargs):
        if title is None:
            title = r"$|g|^2$"
        name = "g2" + name_postfix

        self.plot_mesh(self.mesh.g2, name=name, title=title, **kwargs)

    def psi2(self, name_postfix: str = "", **kwargs):
        title = r"$|\Psi|^2$"
        name = "psi2" + name_postfix

        self.plot_mesh(self.mesh.psi2, name=name, title=title, **kwargs)


class LineMeshPlotter(MeshPlotter):
    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: u.Unit = "bohr_radius",
        norm=si.vis.AbsoluteRenormalize(),
        plot_limit=None,
        slicer="get_mesh_slicer",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        (line,) = axis.plot(
            self.mesh.z_mesh[_slice] / unit_value, norm(mesh[_slice]), **kwargs
        )

        return line

    def plot_mesh(
        self, mesh: "meshes.ScalarMesh", distance_unit: u.Unit = "nm", **kwargs
    ):
        si.vis.xy_plot(
            self.sim.name + "_" + kwargs.pop("name"),
            self.mesh.z_mesh,
            mesh,
            x_label="Distance $x$",
            x_unit=distance_unit,
            **kwargs,
        )

    def update_mesh(self, colormesh, updated_mesh, norm=None, **kwargs):
        if norm is not None:
            updated_mesh = norm(updated_mesh)

        super().update_mesh(colormesh, updated_mesh, **kwargs)


class RectangleMeshPlotter(MeshPlotter):
    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: u.Unit = "nm",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer="get_mesh_slicer",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.mesh.x_mesh[_slice] / unit_value,
            self.mesh.z_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )

        return color_mesh

    #
    # def attach_probability_current_to_axis(
    #     self,
    #     axis: plt.Axes,
    #     plot_limit: Optional[float] = None,
    #     distance_unit: u.Unit = 'bohr_radius',
    #     rate_unit = 'per_asec',
    # ):
    #     distance_unit_value, _ = u.get_unit_value_and_latex(distance_unit)
    #     rate_unit_value, _ = u.get_unit_value_and_latex(rate_unit)
    #
    #     current_mesh_z, current_mesh_rho = self.mesh.get_probability_current_density_vector_field()  # actually densities here
    #
    #     current_mesh_z *= self.mesh.delta_z
    #     current_mesh_rho *= self.mesh.delta_rho
    #
    #     skip_count = int(self.mesh.z_mesh.shape[0] / 50), int(self.mesh.z_mesh.shape[1] / 50)
    #     skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))
    #
    #     normalization = np.nanmax(np.sqrt((current_mesh_z ** 2) + (current_mesh_rho ** 2))[skip])
    #     if normalization == 0:
    #         normalization = 1
    #
    #     sli = self.mesh.get_mesh_slicer(plot_limit)
    #
    #     quiv = axis.quiver(
    #         self.mesh.z_mesh[sli][skip] / distance_unit_value,
    #         self.mesh.rho_mesh[sli][skip] / distance_unit_value,
    #         current_mesh_z[sli][skip] / normalization,
    #         current_mesh_rho[sli][skip] / normalization,
    #         pivot = 'middle',
    #         scale = 10,
    #         units = 'width',
    #         scale_units = 'width',
    #         alpha = 0.5,
    #         color = 'white',
    #     )
    #
    #     return quiv

    def plot_mesh(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: u.Unit = "nm",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit=None,
        slicer="get_mesh_slicer",
        show_colorbar=True,
        show_title=True,
        show_axes=True,
        grid_kwargs=None,
        overlay_probability_current=False,
        **kwargs,
    ):
        grid_kwargs = collections.ChainMap(
            grid_kwargs or {}, si.vis.DEFAULT_COLORMESH_GRID_KWARGS
        )
        unit_value, unit_name = u.get_unit_value_and_latex(distance_unit)

        with si.vis.FigureManager(f"{self.spec.name}__{name}", **kwargs) as figman:
            fig = figman.fig
            ax = plt.subplot(111)

            color_mesh = self.attach_mesh_to_axis(
                ax,
                mesh,
                distance_unit=distance_unit,
                colormap=colormap,
                norm=norm,
                shading=shading,
                plot_limit=plot_limit,
                slicer=slicer,
            )

            ax.set_xlabel(rf"$x$ (${unit_name}$)")
            ax.set_ylabel(rf"$z$ (${unit_name}$)")
            if title is not None and title != "" and show_axes and show_title:
                ax.set_title(title, y=1.1)

            if show_colorbar and show_axes:
                cax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
                plt.colorbar(mappable=color_mesh, cax=cax)

            # if overlay_probability_current:
            #     self.attach_probability_current_to_axis(ax)

            ax.axis("tight")  # removes blank space between color mesh and axes

            ax.grid(
                True, color=si.vis.CMAP_TO_OPPOSITE[colormap], **grid_kwargs
            )  # change grid color to make it show up against the colormesh

            ax.tick_params(labelright=True, labeltop=True)  # ticks on all sides

            if not show_axes:
                ax.axis("off")

        return figman


class CylindricalSliceMeshPlotter(MeshPlotter):
    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: u.Unit = "bohr_radius",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer="get_mesh_slicer",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.mesh.z_mesh[_slice] / unit_value,
            self.mesh.rho_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )

        return color_mesh

    def attach_probability_current_to_axis(
        self,
        axis: plt.Axes,
        plot_limit: Optional[float] = None,
        distance_unit: u.Unit = "bohr_radius",
        rate_unit="per_asec",
    ):
        distance_unit_value, _ = u.get_unit_value_and_latex(distance_unit)
        rate_unit_value, _ = u.get_unit_value_and_latex(rate_unit)

        (
            current_mesh_z,
            current_mesh_rho,
        ) = (
            self.mesh.get_probability_current_density_vector_field()
        )  # actually densities here

        current_mesh_z *= self.mesh.delta_z
        current_mesh_rho *= self.mesh.delta_rho

        skip_count = (
            int(self.mesh.z_mesh.shape[0] / 50),
            int(self.mesh.z_mesh.shape[1] / 50),
        )
        skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))

        normalization = np.nanmax(
            np.sqrt((current_mesh_z ** 2) + (current_mesh_rho ** 2))[skip]
        )
        if normalization == 0:
            normalization = 1

        sli = self.mesh.get_mesh_slicer(plot_limit)

        quiv = axis.quiver(
            self.mesh.z_mesh[sli][skip] / distance_unit_value,
            self.mesh.rho_mesh[sli][skip] / distance_unit_value,
            current_mesh_z[sli][skip] / normalization,
            current_mesh_rho[sli][skip] / normalization,
            pivot="middle",
            scale=10,
            units="width",
            scale_units="width",
            alpha=0.5,
            color="white",
        )

        return quiv

    def plot_mesh(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: u.Unit = "bohr_radius",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit=None,
        slicer="get_mesh_slicer",
        show_colorbar=True,
        show_title=True,
        show_axes=True,
        grid_kwargs=None,
        overlay_probability_current=False,
        **kwargs,
    ):
        grid_kwargs = collections.ChainMap(
            grid_kwargs or {}, si.vis.COLORMESH_GRID_KWARGS
        )
        unit_value, unit_name = u.get_unit_value_and_latex(distance_unit)

        with si.vis.FigureManager(f"{self.spec.name}__{name}", **kwargs) as figman:
            fig = figman.fig
            ax = plt.subplot(111)

            color_mesh = self.attach_mesh_to_axis(
                ax,
                mesh,
                distance_unit=distance_unit,
                colormap=colormap,
                norm=norm,
                shading=shading,
                plot_limit=plot_limit,
                slicer=slicer,
            )

            ax.set_xlabel(rf"$z$ (${unit_name}$)")
            ax.set_ylabel(rf"$\rho$ (${unit_name}$)")
            if title is not None and title != "" and show_axes and show_title:
                ax.set_title(title, y=1.1)

            if show_colorbar and show_axes:
                cax = fig.add_axes([1.0, 0.1, 0.02, 0.8])
                plt.colorbar(mappable=color_mesh, cax=cax)

            if overlay_probability_current:
                self.attach_probability_current_to_axis(ax)

            ax.axis("tight")  # removes blank space between color mesh and axes

            ax.grid(
                True, color=si.vis.CMAP_TO_OPPOSITE[colormap], **grid_kwargs
            )  # change grid color to make it show up against the colormesh

            ax.tick_params(labelright=True, labeltop=True)  # ticks on all sides

            if not show_axes:
                ax.axis("off")

        return figman


def fmt_polar_axis(fig, axis, colormap, grid_kwargs, unit_latex):
    axis.grid(
        True, color=si.vis.CMAP_TO_OPPOSITE[colormap], **grid_kwargs
    )  # change grid color to make it show up against the colormesh
    angle_labels = [
        f"{s}\u00b0" for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)
    ]  # \u00b0 is unicode degree symbol
    axis.set_thetagrids(np.arange(0, 359, 30), frac=1.075, labels=angle_labels)

    axis.tick_params(
        axis="both", which="major", labelsize=6
    )  # increase size of tick labels
    axis.tick_params(
        axis="y", which="major", colors=si.vis.CMAP_TO_OPPOSITE[colormap], pad=3
    )  # make r ticks a color that shows up against the colormesh
    axis.tick_params(axis="both", which="both", length=0)

    axis.set_rlabel_position(80)

    max_yticks = 5
    yloc = plt.MaxNLocator(max_yticks, symmetric=False, prune="both")
    axis.yaxis.set_major_locator(yloc)

    fig.canvas.draw()  # must draw early to modify the axis text

    tick_labels = axis.get_yticklabels()
    for t in tick_labels:
        t.set_text(t.get_text() + rf"${unit_latex}$")
        axis.set_yticklabels(tick_labels)


class SphericalSliceMeshPlotter(MeshPlotter):
    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: u.Unit = "bohr_radius",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer="get_mesh_slicer",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.mesh.theta_mesh[_slice],
            self.mesh.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )
        color_mesh_mirror = axis.pcolormesh(
            -self.mesh.theta_mesh[_slice],
            self.mesh.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )  # another colormesh, mirroring the first mesh onto pi to 2pi

        return color_mesh, color_mesh_mirror

    def attach_probability_current_to_axis(
        self, axis, plot_limit=None, distance_unit: u.Unit = "bohr_radius"
    ):
        raise NotImplementedError

    def plot_mesh(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: u.Unit = "bohr_radius",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit=None,
        slicer="get_mesh_slicer",
        show_colorbar=True,
        show_title=True,
        show_axes=True,
        grid_kwargs=None,
        overlay_probability_current=False,
        **kwargs,
    ):
        grid_kwargs = collections.ChainMap(
            grid_kwargs or {}, si.vis.COLORMESH_GRID_KWARGS
        )
        unit_value, unit_latex = u.get_unit_value_and_latex(distance_unit)

        with si.vis.FigureManager(f"{self.spec.name}__{name}", **kwargs) as figman:
            fig = figman.fig
            fig.set_tight_layout(True)
            ax = plt.subplot(111, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction("clockwise")

            color_mesh, color_mesh_mirror = self.attach_mesh_to_axis(
                ax,
                mesh,
                distance_unit=distance_unit,
                colormap=colormap,
                norm=norm,
                shading=shading,
                plot_limit=plot_limit,
                slicer=slicer,
            )

            if title is not None:
                title = ax.set_title(title, fontsize=15)
                title.set_x(0.03)  # move title to the upper left corner
                title.set_y(0.97)

            if show_colorbar and show_axes:
                cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
                plt.colorbar(mappable=color_mesh, cax=cax)

            fmt_polar_axis(fig, ax, colormap, grid_kwargs, unit_latex)

            ax.set_rmax((self.mesh.r_max - (self.mesh.delta_r / 2)) / unit_value)


class SphericalHarmonicMeshPlotter(MeshPlotter):
    def attach_mesh_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: str = "bohr_radius",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer: str = "get_mesh_slicer_spatial",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.mesh.theta_plot_mesh[_slice],
            self.mesh.r_theta_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )

        return color_mesh

    def plot_mesh(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: u.Unit = "bohr_radius",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit=None,
        slicer="get_mesh_slicer",
        show_colorbar=True,
        show_title=True,
        show_axes=True,
        grid_kwargs=None,
        overlay_probability_current=False,
        **kwargs,
    ):
        grid_kwargs = collections.ChainMap(
            grid_kwargs or {}, si.vis.COLORMESH_GRID_KWARGS
        )
        unit_value, unit_latex = u.get_unit_value_and_latex(distance_unit)

        with si.vis.FigureManager(name=f"{self.spec.name}__{name}", **kwargs) as figman:
            fig = figman.fig
            fig.set_tight_layout(True)
            ax = plt.subplot(111, projection="polar")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction("clockwise")

            color_mesh = self.attach_mesh_to_axis(
                ax,
                mesh,
                distance_unit=distance_unit,
                colormap=colormap,
                norm=norm,
                shading=shading,
                plot_limit=plot_limit,
                slicer=slicer,
            )
            if title is not None and title != "" and show_axes and show_title:
                title = ax.set_title(title, fontsize=15)
                title.set_x(0.03)  # move title to the upper left corner
                title.set_y(0.97)

            if show_colorbar and show_axes:
                cax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
                plt.colorbar(mappable=color_mesh, cax=cax)

            fmt_polar_axis(fig, ax, colormap, grid_kwargs, unit_latex)

            if plot_limit is not None and plot_limit < self.mesh.r_max:
                ax.set_rmax((plot_limit - (self.mesh.delta_r / 2)) / unit_value)
            else:
                ax.set_rmax((self.mesh.r_max - (self.mesh.delta_r / 2)) / unit_value)

            if not show_axes:
                ax.axis("off")

    def attach_g_to_axis(
        self, axis: plt.Axes, colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):
        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(
                    np.abs(self.mesh.space_g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
                )
            )

        return self.attach_mesh_to_axis(
            axis, self.mesh.space_g, colormap=colormap, norm=norm, **kwargs
        )

    def g(
        self,
        name_postfix: str = "",
        colormap=plt.get_cmap("richardson"),
        norm=None,
        **kwargs,
    ):
        title = r"$g$"
        name = "g" + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(
                    np.abs(self.mesh.space_g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
                )
            )

        self.plot_mesh(
            self.mesh.space_g,
            name=name,
            title=title,
            colormap=colormap,
            norm=norm,
            show_colorbar=False,
            **kwargs,
        )

    def attach_psi_to_axis(
        self, axis: plt.Axes, colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):
        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(
                    np.abs(self.mesh.space_psi)
                    / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
                )
            )

        return self.attach_mesh_to_axis(
            axis, self.mesh.space_psi, colormap=colormap, norm=norm, **kwargs
        )

    def psi(
        self,
        name_postfix: str = "",
        colormap=plt.get_cmap("richardson"),
        norm=None,
        **kwargs,
    ):
        title = r"$\Psi$"
        name = "psi" + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(
                    np.abs(self.mesh.space_psi)
                    / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR
                )
            )

        self.plot_mesh(
            self.mesh.space_psi,
            name=name,
            title=title,
            colormap=colormap,
            norm=norm,
            show_colorbar=False,
            **kwargs,
        )

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.space_g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.mesh.space_psi, **kwargs)

    # I have no idea what this method does, sinec it doesn't use mesh...
    def attach_mesh_repr_to_axis(
        self,
        axis: plt.Axes,
        mesh: "meshes.ScalarMesh",
        distance_unit: str = "bohr_radius",
        colormap=plt.get_cmap("inferno"),
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer: str = "get_mesh_slicer",
        **kwargs,
    ):
        unit_value, _ = u.get_unit_value_and_latex(distance_unit)

        _slice = getattr(self.mesh, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(
            self.mesh.l_mesh[_slice],
            self.mesh.r_mesh[_slice] / unit_value,
            mesh[_slice],
            shading=shading,
            cmap=colormap,
            norm=norm,
            **kwargs,
        )

        return color_mesh

    def plot_mesh_repr(
        self,
        mesh: "meshes.ScalarMesh",
        name: str = "",
        title: Optional[str] = None,
        distance_unit: str = "bohr_radius",
        colormap=vis.COLORMAP_WAVEFUNCTION,
        norm=si.vis.AbsoluteRenormalize(),
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        plot_limit: Optional[float] = None,
        slicer: str = "get_mesh_slicer",
        aspect_ratio: float = si.vis.GOLDEN_RATIO,
        show_colorbar: bool = True,
        show_title: bool = True,
        show_axes: bool = True,
        title_y_adjust: float = 1.1,
        title_size: float = 12,
        axis_label_size: float = 12,
        tick_label_size: float = 10,
        grid_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if grid_kwargs is None:
            grid_kwargs = {}
        with si.vis.FigureManager(
            name=f"{self.spec.name}__{name}", aspect_ratio=aspect_ratio, **kwargs
        ) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)
            axis = plt.subplot(111)

            unit_value, unit_latex = u.get_unit_value_and_latex(distance_unit)

            color_mesh = self.attach_mesh_repr_to_axis(
                axis,
                mesh,
                distance_unit=distance_unit,
                colormap=colormap,
                norm=norm,
                shading=shading,
                plot_limit=plot_limit,
                slicer=slicer,
            )

            axis.set_xlabel(r"$\ell$", fontsize=axis_label_size)
            axis.set_ylabel(rf"$r$ (${unit_latex}$)", fontsize=axis_label_size)
            if title is not None and title != "" and show_axes and show_title:
                title = axis.set_title(title, fontsize=title_size)
                title.set_y(title_y_adjust)  # move title up a bit

            # make a colorbar
            if show_colorbar and show_axes:
                cbar = fig.colorbar(mappable=color_mesh, ax=axis)
                cbar.ax.tick_params(labelsize=tick_label_size)

            axis.grid(
                True,
                color=si.vis.CMAP_TO_OPPOSITE[colormap.name],
                **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs},
            )  # change grid color to make it show up against the colormesh

            axis.tick_params(labelright=True, labeltop=True)  # ticks on all sides
            axis.tick_params(
                axis="both", which="major", labelsize=tick_label_size
            )  # increase size of tick labels
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            y_ticks = axis.yaxis.get_major_ticks()
            y_ticks[0].label1.set_visible(False)
            y_ticks[0].label2.set_visible(False)
            y_ticks[-1].label1.set_visible(False)
            y_ticks[-1].label2.set_visible(False)

            axis.axis("tight")

            if not show_axes:
                axis.axis("off")

    def g_repr(
        self,
        name_postfix: str = "",
        title: Optional[str] = None,
        colormap=plt.get_cmap("richardson"),
        norm=None,
        **kwargs,
    ):
        if title is None:
            title = r"$g$"
        name = "g_repr" + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        self.plot_mesh_repr(
            self.mesh.g,
            name=name,
            title=title,
            colormap=colormap,
            norm=norm,
            show_colorbar=False,
            **kwargs,
        )

    def attach_g_repr_to_axis(
        self, axis: plt.Axes, colormap=plt.get_cmap("richardson"), norm=None, **kwargs
    ):

        if norm is None:
            norm = si.vis.RichardsonNormalization(
                np.max(np.abs(self.mesh.g) / vis.DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR)
            )

        return self.attach_mesh_repr_to_axis(
            axis, self.mesh.g, colormap=colormap, norm=norm, **kwargs
        )

    def electron_momentum_spectrum(
        self,
        r_type: u.Unit = "wavenumber",
        r_scale: u.Unit = "per_nm",
        r_lower_lim: float = u.twopi * 0.01 * u.per_nm,
        r_upper_lim: float = u.twopi * 10 * u.per_nm,
        r_points: int = 100,
        theta_points: int = 360,
        g: Optional["meshes.StateOrGMesh"] = None,
        **kwargs,
    ):
        if r_type not in ("wavenumber", "energy", "momentum"):
            raise ValueError(
                "Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'"
            )

        thetas = np.linspace(0, u.twopi, theta_points)
        r = np.linspace(r_lower_lim, r_upper_lim, r_points)

        if r_type == "wavenumber":
            wavenumbers = r
        elif r_type == "energy":
            wavenumbers = core.electron_wavenumber_from_energy(r)
        elif r_type == "momentum":
            wavenumbers = r / u.hbar

        g = self.mesh.state_to_g(g)

        (
            theta_mesh,
            wavenumber_mesh,
            inner_product_mesh,
        ) = self.mesh.inner_product_with_plane_waves(thetas, wavenumbers, g=g)

        if r_type == "wavenumber":
            r_mesh = wavenumber_mesh
        elif r_type == "energy":
            r_mesh = core.electron_energy_from_wavenumber(wavenumber_mesh)
        elif r_type == "momentum":
            r_mesh = wavenumber_mesh * u.hbar

        return self.plot_electron_momentum_spectrum_from_meshes(
            theta_mesh, r_mesh, inner_product_mesh, r_type, r_scale, **kwargs
        )

    def electron_momentum_spectrum_from_meshes(
        self,
        theta_mesh,
        r_mesh,
        inner_product_mesh,
        r_type: str,
        r_scale: float,
        log: bool = False,
        shading: si.vis.ColormapShader = si.vis.ColormapShader.FLAT,
        **kwargs,
    ):
        """
        Generate a polar plot of the wavefunction decomposed into plane waves.

        The radial dimension can be displayed in wavenumbers, energy, or momentum. The angle is the angle of the plane wave in the z-x plane (because m=0, the decomposition is symmetric in the x-y plane).

        :param r_type: type of unit for the radial axis ('wavenumber', 'energy', or 'momentum')
        :param r_scale: unit specification for the radial dimension
        :param r_lower_lim: lower limit for the radial dimension
        :param r_upper_lim: upper limit for the radial dimension
        :param r_points: number of points for the radial dimension
        :param theta_points: number of points for the angular dimension
        :param log: True to displayed logged data, False otherwise (default: False)
        :param kwargs: kwargs are passed to compy.utils.FigureManager
        :return: the FigureManager generated during plot creation
        """
        if r_type not in ("wavenumber", "energy", "momentum"):
            raise ValueError(
                "Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'"
            )

        r_unit_value, r_unit_name = u.get_unit_value_and_latex(r_scale)

        plot_kwargs = {**dict(aspect_ratio=1), **kwargs}

        r_mesh = np.real(r_mesh)
        overlap_mesh = np.abs(inner_product_mesh) ** 2

        with si.vis.FigureManager(
            self.sim.name + "__electron_spectrum", **plot_kwargs
        ) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)

            axis = plt.subplot(111, projection="polar")
            axis.set_theta_zero_location("N")
            axis.set_theta_direction("clockwise")

            figman.name += f"__{r_type}"

            norm = None
            if log:
                norm = matplotlib.colors.LogNorm(
                    vmin=np.nanmin(overlap_mesh), vmax=np.nanmax(overlap_mesh)
                )
                figman.name += "__log"

            color_mesh = axis.pcolormesh(
                theta_mesh,
                r_mesh / r_unit_value,
                overlap_mesh,
                shading=shading,
                norm=norm,
                cmap="viridis",
            )

            # make a colorbar
            cbar_axis = fig.add_axes(
                [1.01, 0.1, 0.04, 0.8]
            )  # add a new axis for the cbar so that the old axis can stay square
            cbar = plt.colorbar(mappable=color_mesh, cax=cbar_axis)
            cbar.ax.tick_params(labelsize=10)

            axis.grid(
                True,
                color=si.vis.COLOR_OPPOSITE_VIRIDIS,
                **si.vis.COLORMESH_GRID_KWARGS,
            )  # change grid color to make it show up against the colormesh
            angle_labels = [
                f"{s}\u00b0"
                for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)
            ]  # \u00b0 is unicode degree symbol
            axis.set_thetagrids(np.arange(0, 359, 30), frac=1.075, labels=angle_labels)

            axis.tick_params(
                axis="both", which="major", labelsize=8
            )  # increase size of tick labels
            axis.tick_params(
                axis="y", which="major", colors=si.vis.COLOR_OPPOSITE_VIRIDIS, pad=3
            )  # make r ticks a color that shows up against the colormesh
            axis.tick_params(axis="both", which="both", length=0)

            axis.set_rlabel_position(80)

            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric=False, prune="both")
            axis.yaxis.set_major_locator(yloc)

            fig.canvas.draw()  # must draw early to modify the axis text

            tick_labels = axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + rf"${r_unit_name}$")
            axis.set_yticklabels(tick_labels)

            axis.set_rmax(np.nanmax(r_mesh) / r_unit_value)

        return figman
