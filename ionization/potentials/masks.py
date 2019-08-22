import functools

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import summables, utils, exceptions


class Mask(summables.Summand):
    """A class representing a spatial 'mask' that can be applied to the wavefunction to reduce it in certain regions."""

    def __init__(self):
        super().__init__()
        self.summation_class = MaskSum


class MaskSum(summables.Sum, Mask):
    """A class representing a combination of masks."""

    container_name = "masks"

    def __call__(self, *args, **kwargs):
        # masks should be multiplied together, not summed
        return functools.reduce(
            lambda a, b: a * b, (x(*args, **kwargs) for x in self._container)
        )


class NoMask(Mask):
    """A class representing the lack of a mask."""

    def __call__(self, *args, **kwargs):
        return 1


class RadialCosineMask(Mask):
    """A mask which begins at some radius and smoothly decreases to 0 as the :math:`n` th-root of cosine."""

    def __init__(
        self,
        inner_radius: float = 50 * u.bohr_radius,
        outer_radius: float = 100 * u.bohr_radius,
        smoothness: float = 8,
    ):
        """
        Parameters
        ----------
        inner_radius
            The inner radius of the mask.
            The mask evaluates to ``1`` for the last time here.
        outer_radius
            The outer radius of the mask.
            The mask evaluates to ``0`` for the first time here.
        smoothness
            The inverse of the power to which the cosine will be taken.
        """
        if inner_radius < 0 or outer_radius < 0:
            raise exceptions.InvalidMaskParameter(
                "inner and outer radius must be non-negative"
            )
        if inner_radius >= outer_radius:
            raise exceptions.InvalidMaskParameter(
                "outer radius must be larger than inner radius"
            )
        if smoothness < 1:
            raise exceptions.InvalidMaskParameter("smoothness must be greater than 1")

        super().__init__()

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.smoothness = smoothness

    def __call__(self, *, r, **kwargs):
        return np.where(
            np.greater_equal(r, self.inner_radius) * np.less(r, self.outer_radius),
            np.abs(
                np.cos(
                    0.5
                    * u.pi
                    * (r - self.inner_radius)
                    / np.abs(self.outer_radius - self.inner_radius)
                )
            )
            ** (1 / self.smoothness),
            np.where(np.greater_equal(r, self.outer_radius), 0, 1),
        )

    def __repr__(self):
        return utils.make_repr(self, "inner_radius", "outer_radius", "smoothness")

    def __str__(self):
        return utils.make_repr(
            self,
            ("inner_radius", "bohr_radius"),
            ("outer_radius", "bohr_radius"),
            "smoothness",
        )

    def info(self) -> si.Info:
        info = super().info()

        info.add_field(
            "Inner Radius", utils.fmt_quantity(self.inner_radius, utils.LENGTH_UNITS)
        )
        info.add_field(
            "Outer Radius", utils.fmt_quantity(self.outer_radius, utils.LENGTH_UNITS)
        )
        info.add_field("Smoothness", self.smoothness)

        return info
