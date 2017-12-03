import scipy.interpolate as interp
import scipy.integrate as integ


class InterpolatedIntegrator:
    def __init__(self, interpolator = interp.CubicSpline, integrator = integ.quadrature):
        self.interpolator = interpolator
        self.integrator = integrator

    def __call__(self, y, x):
        lower_bound = x[0]
        upper_bound = x[-1]

        interpolated_y = self.interpolator(y = y, x = x)

        # integral = interpolated_y.integrate(lower_bound, upper_bound)
        integral = self.integrator(interpolated_y, lower_bound, upper_bound)

        return integral
