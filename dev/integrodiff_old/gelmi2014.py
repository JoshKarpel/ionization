import numpy as np
import scipy as sp
import scipy.integrate as integ
import scipy.optimize as optim
import scipy.interpolate as inter

import matplotlib.pyplot as plt

import simulacra as si


class IDESolver:
    def __init__(self, y0, x, c, d, k, alpha, beta, F, tol=1e-8):
        self.x = x
        self.c = c
        self.d = d
        self.k = k
        self.F = F

        self.y0 = y0

        self.tol = tol

    def initial_guess(self):
        return integ.odeint(self.c, self.y0, self.x)[:, 0]

    def compare(self, y1, y2):
        diff = y1 - y2
        return np.sum(diff * diff)

    def solve_rhs_with_known_y(self, y):
        s = self.x
        y = inter.interp1d(self.x, y, fill_value="extrapolate")
        return integ.odeint(
            lambda _, x: self.c(y(x), x)
            + (self.d(x) * integ.simps(y=self.k(x, s) * self.F(y(s)), x=s)),
            self.y0,
            self.x,
        )[:, 0]

    def next_curr(self, curr, guess):
        a = 0.5
        return (a * curr) + ((1 - a) * guess)

    def solve(self):
        curr = self.initial_guess()
        guess = self.solve_rhs_with_known_y(curr)

        c = 0

        while self.compare(curr, guess) > self.tol:
            print(self.compare(curr, guess))
            curr = self.next_curr(curr, guess)
            guess = self.solve_rhs_with_known_y(curr)

            c += 1
            print(c)
            # if c == 10: break

        return guess


if __name__ == "__main__":
    s = IDESolver(
        y0=0,
        x=np.linspace(0, 1, 100),
        c=lambda y, x: y - (0.5 * x) + (1 / (1 + x)) - np.log(1 + x),
        d=lambda x: 1 / (np.log(2)) ** 2,
        k=lambda x, s: x / (1 + s),
        alpha=lambda x: 0,
        beta=lambda x: 1,
        F=lambda y: y,
    )

    # print(s.initial_guess())
    # s.solve()

    soln = s.solve()
    exact = np.log(1 + s.x)

    si.vis.xy_plot(
        "init",
        s.x,
        s.initial_guess(),
        soln,
        exact,
        line_labels=["init guess", "solve", "solution"],
        save_on_exit=False,
        show=True,
    )

    si.vis.xy_plot(
        "err", s.x, np.abs(exact - soln), y_log_axis=True, save_on_exit=False, show=True
    )
