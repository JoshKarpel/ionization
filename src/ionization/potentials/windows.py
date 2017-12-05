import functools

import numpy as np

import simulacra as si
import simulacra.units as u


class TimeWindow(si.Summand):
    """A class representing a time-window that can be attached to another potential."""

    def __init__(self):
        super().__init__()
        self.summation_class = TimeWindowSum


class TimeWindowSum(si.Sum, TimeWindow):
    """A class representing a combination of time-windows."""

    container_name = 'windows'

    def __call__(self, *args, **kwargs):
        return functools.reduce(lambda a, b: a * b, (x(*args, **kwargs) for x in self._container))  # windows should be multiplied together, not summed


class NoTimeWindow(TimeWindow):
    """A class representing the lack of a time-window."""

    def __call__(self, t):
        return 1


class RectangularTimeWindow(TimeWindow):
    def __init__(self, on_time = 0 * u.asec, off_time = 50 * u.asec):
        self.on_time = on_time
        self.off_time = off_time

        super().__init__()

    def __str__(self):
        return '{}(on at = {} as, off at = {} as)'.format(self.__class__.__name__,
                                                          u.uround(self.on_time, u.asec),
                                                          u.uround(self.off_time, u.asec))

    def __repr__(self):
        return '{}(on_time = {}, off_time = {})'.format(self.__class__.__name__,
                                                        self.on_time,
                                                        self.off_time)

    def __call__(self, t):
        cond = np.greater_equal(t, self.on_time) * np.less_equal(t, self.off_time)
        on = 1
        off = 0

        return np.where(cond, on, off)

    def info(self):
        info = super().info()

        info.add_field('Time On', f'{u.uround(self.on_time, u.asec)} u.asec')
        info.add_field('Time Off', f'{u.uround(self.off_time, u.asec)} u.asec')

        return info


class LinearRampTimeWindow(TimeWindow):
    def __init__(self, ramp_on_time = 0 * u.asec, ramp_time = 50 * u.asec):
        self.ramp_on_time = ramp_on_time
        self.ramp_time = ramp_time

        # TODO: ramp_from and ramp_to

        super().__init__()

    def __str__(self):
        return '{}(ramp on at = {} as, ramp time = {} as)'.format(self.__class__.__name__,
                                                                  u.uround(self.ramp_on_time, u.asec),
                                                                  u.uround(self.ramp_time, u.asec))

    def __repr__(self):
        return '{}(ramp_on_time = {}, ramp_time = {})'.format(self.__class__.__name__,
                                                              self.ramp_on_time,
                                                              self.ramp_time)

    def __call__(self, t):
        cond = np.greater_equal(t, self.ramp_on_time)
        on = 1
        off = 0

        out_1 = np.where(cond, on, off)

        cond = np.less_equal(t, self.ramp_on_time + self.ramp_time)
        on = np.ones(np.shape(t)) * (t - self.ramp_on_time) / self.ramp_time
        off = 1

        out_2 = np.where(cond, on, off)

        return out_1 * out_2

    def info(self):
        info = super().info()

        info.add_field('Time Ramp Start', f'{u.uround(self.ramp_on_time, u.asec)} u.asec')

        return info


class SymmetricExponentialTimeWindow(TimeWindow):
    def __init__(self, window_time = 500 * u.asec, window_width = 10 * u.asec, window_center = 0 * u.asec):
        self.window_time = window_time
        self.window_width = window_width
        self.window_center = window_center

        super().__init__()

    def __str__(self):
        return '{}(window time = {} as, window width = {} as, window center = {})'.format(self.__class__.__name__,
                                                                                          u.uround(self.window_time, u.asec),
                                                                                          u.uround(self.window_width, u.asec),
                                                                                          u.uround(self.window_center, u.asec))

    def __repr__(self):
        return '{}(window_time = {}, window_width = {}, window_center = {})'.format(self.__class__.__name__,
                                                                                    self.window_time,
                                                                                    self.window_width,
                                                                                    self.window_center)

    def __call__(self, t):
        tau = np.array(t) - self.window_center
        return np.abs(1 / (1 + np.exp(-(tau + self.window_time) / self.window_width)) - 1 / (1 + np.exp(-(tau - self.window_time) / self.window_width)))

    def info(self):
        info = super().info()

        info.add_field('Window Time', f'{u.uround(self.window_time, u.asec)} as | {u.uround(self.window_time, u.fsec)} fs | {u.uround(self.window_time, u.atomic_time)} a.u.')
        info.add_field('Window Width', f'{u.uround(self.window_width, u.asec)} as | {u.uround(self.window_width, u.fsec)} fs | {u.uround(self.window_width, u.atomic_time)} a.u.')
        info.add_field('Window Center', f'{u.uround(self.window_center, u.asec)} as | {u.uround(self.window_center, u.fsec)} fs | {u.uround(self.window_center, u.atomic_time)} a.u.')

        return info


class SmoothedTrapezoidalWindow(TimeWindow):
    def __init__(self, *, time_front, time_plateau):
        super().__init__()

        self.time_front = time_front
        self.time_plateau = time_plateau

    def __call__(self, t):
        cond_before = np.less(t, self.time_front)
        cond_middle = np.less_equal(self.time_front, t) * np.less_equal(t, self.time_front + self.time_plateau)
        cond_after = np.less(self.time_front + self.time_plateau, t) * np.less_equal(t, (2 * self.time_front) + self.time_plateau)

        out = np.where(cond_before, np.sin(u.pi * t / (2 * self.time_front)) ** 2, 0)
        out += np.where(cond_middle, 1, 0)
        out += np.where(cond_after, np.cos(u.pi * (t - (self.time_front + self.time_plateau)) / (2 * self.time_front)) ** 2, 0)

        return out

    def info(self):
        info = super().info()

        info.add_field('Front Time', f'{u.uround(self.time_front, u.asec)} as | {u.uround(self.time_front, u.fsec)} fs | {u.uround(self.time_front, u.atomic_time)} a.u.')
        info.add_field('Plateau Time', f'{u.uround(self.time_plateau, u.asec)} as | {u.uround(self.time_plateau, u.fsec)} fs  | {u.uround(self.time_plateau, u.atomic_time)} a.u.')

        return info
