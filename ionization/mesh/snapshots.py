import logging

import numpy as np

import simulacra as si
import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Snapshot:
    def __init__(self, simulation, time_index):
        self.sim = simulation
        self.spec = self.sim.spec
        self.time_index = time_index

        self.data = dict()

    @property
    def time(self):
        return self.sim.times[self.time_index]

    def __str__(self):
        return f"Snapshot of {self} at time {u.uround(self.sim.times[self.time_index])} as (time index = {self.time_index})"

    def __repr__(self):
        return si.utils.field_str(self, "sim", "time_index")

    def take_snapshot(self):
        self.collect_norm()

    def collect_norm(self):
        self.data["norm"] = self.sim.mesh.norm()


class SphericalHarmonicSnapshot(Snapshot):
    def __init__(
        self,
        simulation,
        time_index,
        plane_wave_overlap__max_wavenumber=50 * u.per_nm,
        plane_wave_overlap__wavenumber_points=500,
        plane_wave_overlap__theta_points=200,
    ):
        super().__init__(simulation, time_index)

        self.plane_wave_overlap__max_wavenumber = plane_wave_overlap__max_wavenumber
        self.plane_wave_overlap__wavenumber_points = (
            plane_wave_overlap__wavenumber_points
        )
        self.plane_wave_overlap__theta_points = plane_wave_overlap__theta_points

    def take_snapshot(self):
        super().take_snapshot()

        for free_only in (True, False):
            self.collect_inner_product_with_plane_waves(free_only=free_only)

    def collect_inner_product_with_plane_waves(self, free_only=False):
        thetas = np.linspace(0, u.twopi, self.plane_wave_overlap__theta_points)
        wavenumbers = np.delete(
            np.linspace(
                0,
                self.plane_wave_overlap__max_wavenumber,
                self.plane_wave_overlap__wavenumber_points + 1,
            ),
            0,
        )

        if free_only:
            key = "inner_product_with_plane_waves__free_only"
            g = self.sim.mesh.get_g_with_states_removed(self.sim.bound_states)
        else:
            key = "inner_product_with_plane_waves"
            g = None

        self.data[key] = self.sim.mesh.inner_product_with_plane_waves(
            thetas, wavenumbers, g=g
        )
