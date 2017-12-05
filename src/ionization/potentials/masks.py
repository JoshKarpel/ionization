import functools

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import exceptions


class Mask(si.Summand):
    """A class representing a spatial 'mask' that can be applied to the wavefunction to reduce it in certain regions."""

    def __init__(self):
        super().__init__()
        self.summation_class = MaskSum


class MaskSum(si.Sum, Mask):
    """A class representing a combination of masks."""

    container_name = 'masks'

    def __call__(self, *args, **kwargs):
        return functools.reduce(lambda a, b: a * b, (x(*args, **kwargs) for x in self._container))  # masks should be multiplied together, not summed


class NoMask(Mask):
    """A class representing the lack of a mask."""

    def __call__(self, *args, **kwargs):
        return 1


class RadialCosineMask(Mask):
    """A class representing a masks which begins at some radius and smoothly decreases to 0 as the nth-root of cosine."""

    def __init__(self, inner_radius = 50 * u.bohr_radius, outer_radius = 100 * u.bohr_radius, smoothness = 8):
        """Construct a RadialCosineMask from an inner radius, outer radius, and cosine 'smoothness' (the cosine will be raised to the 1/smoothness power)."""
        if inner_radius < 0 or outer_radius < 0:
            raise exceptions.InvalidMaskParameter('inner and outer radius must be non-negative')
        if inner_radius >= outer_radius:
            raise exceptions.InvalidMaskParameter('outer radius must be larger than inner radius')
        if smoothness < 1:
            raise exceptions.InvalidMaskParameter('smoothness must be greater than 1')

        super().__init__()

        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.smoothness = smoothness

    def __str__(self):
        return '{}(inner radius = {} a_0, outer radius = {} a_0, smoothness = {})'.format(self.__class__.__name__,
                                                                                          u.uround(self.inner_radius, u.bohr_radius, 3),
                                                                                          u.uround(self.outer_radius, u.bohr_radius, 3),
                                                                                          self.smoothness)

    def __repr__(self):
        return '{}(inner_radius = {}, outer_radius = {}, smoothness = {})'.format(self.__class__.__name__,
                                                                                  self.inner_radius,
                                                                                  self.outer_radius,
                                                                                  self.smoothness)

    def __call__(self, *, r, **kwargs):
        """
        Return the value(s) of the mask at radial position(s) r.

        Accepts only keyword arguments.

        :param r: the radial position coordinate
        :param kwargs: absorbs keyword arguments.
        :return: the value(s) of the mask at r
        """
        return np.where(np.greater_equal(r, self.inner_radius) * np.less(r, self.outer_radius),
                        np.abs(np.cos(0.5 * u.pi * (r - self.inner_radius) / np.abs(self.outer_radius - self.inner_radius))) ** (1 / self.smoothness),
                        np.where(np.greater_equal(r, self.outer_radius), 0, 1))

    def info(self):
        info = super().info()

        info.add_field('Inner Radius', f'{u.uround(self.inner_radius, u.bohr_radius, 3)} a_0 | {u.uround(self.inner_radius, u.nm, 3)} nm')
        info.add_field('Outer Radius', f'{u.uround(self.outer_radius, u.bohr_radius, 3)} a_0 | {u.uround(self.outer_radius, u.nm, 3)} nm')
        info.add_field('Smoothness', self.smoothness)

        return info
