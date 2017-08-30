import os
import itertools
import logging

import traits
import numpy as np
import scipy.interpolate as interp
from mayavi import mlab

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib as mpl

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)


def test_rect():
    x_mesh, y_mesh, z_mesh = np.mgrid[-10:10:50j, -5:5:50j, -5:5:50j]
    g_mesh = (x_mesh ** 2) + (y_mesh ** 2) + (z_mesh ** 2)

    # x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    # g_mesh = np.sin(x * y * z) / (x * y * z)

    # scatter = mlab.pipeline.scalar_scatter(x_mesh, y_mesh, z_mesh, g_mesh)
    # scatter = mlab.pipeline.scalar_field(g_mesh)
    # mlab.pipeline.volume(scatter, vmin = 0, vmax = 1)

    # mlab.points3d(x_mesh, y_mesh, z_mesh, g_mesh)
    # mlab.contour3d(g_mesh)
    print(x_mesh.shape)
    print(y_mesh.shape)
    print(z_mesh.shape)
    print(g_mesh.shape)
    mlab.contour3d(x_mesh, y_mesh, z_mesh, g_mesh)
    # mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh))

    mlab.show()


def test_sph():
    r_max = 10
    r = np.linspace(0, r_max, 50)
    theta = np.linspace(0, pi, 50)
    phi = np.linspace(0, twopi, 50)

    # r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, phi, indexing = 'ij')
    r_mesh, theta_mesh = np.meshgrid(r, theta, indexing = 'ij')

    g = (r_mesh ** 2)
    print(r.shape)
    print(theta.shape)
    print(g.shape)

    g_func = interp.RectBivariateSpline(r, theta, g)

    def g_func_func(x_mesh, y_mesh, z_mesh):
        r_mesh = np.sqrt(x_mesh ** 2 + y_mesh ** 2 + z_mesh ** 2)
        print(si.utils.bytes_to_str(r_mesh.nbytes))
        theta_mesh = np.arccos(z_mesh / r_mesh)
        print(si.utils.bytes_to_str(theta_mesh.nbytes))
        return g_func.ev(r_mesh, theta_mesh)

    x_max = 10 / (2 * np.sqrt(3))
    x = np.linspace(-x_max, x_max, 50)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, x, x, indexing = 'ij')

    g_mesh = g_func_func(x_mesh, y_mesh, z_mesh)
    print(si.utils.bytes_to_str(g_mesh.nbytes))

    # x_mesh = r_mesh * np.sin(theta_mesh) * np.cos(phi_mesh)
    # y_mesh = r_mesh * np.sin(theta_mesh) * np.sin(phi_mesh)
    # z_mesh = r_mesh * np.cos(theta_mesh)

    # g_mesh = r_mesh ** 2

    # print(x_mesh)
    # print(y_mesh)
    # print(z_mesh)

    # mlab.contour3d(g_mesh)
    # mlab.contour3d(x_mesh, y_mesh, z_mesh, g_mesh)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh))

    mlab.show()


def test_hyd():
    sim = ion.SphericalHarmonicSpecification(
        'test',
        r_bound = 50 * bohr_radius,
        r_points = 400,
        l_bound = 200,
        theta_points = 360,
        initial_state = ion.HydrogenBoundState(1, 0),
        electric_potential = ion.Rectangle(20 * asec, 80 * asec, 1 * atomic_electric_field),
        time_final = 100 * asec,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    r, theta = sim.mesh.r, sim.mesh.theta

    g_r_theta = np.abs(sim.mesh.space_g)

    g_func = interp.RectBivariateSpline(r, theta, g_r_theta)

    def g_func_func(x_mesh, y_mesh, z_mesh):
        r_mesh = np.sqrt(x_mesh ** 2 + y_mesh ** 2 + z_mesh ** 2)
        print(si.utils.bytes_to_str(r_mesh.nbytes))
        theta_mesh = np.arccos(z_mesh / r_mesh)
        print(si.utils.bytes_to_str(theta_mesh.nbytes))
        return g_func.ev(r_mesh, theta_mesh)

    # x_max = sim.spec.r_bound
    x_max = 20 * bohr_radius
    x = np.linspace(-x_max, x_max, 100)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, x, x, indexing = 'ij')

    g_mesh = g_func_func(x_mesh, y_mesh, z_mesh)
    print(si.utils.bytes_to_str(g_mesh.nbytes))

    mini = g_mesh.min()
    maxi = g_mesh.max()
    # rng = maxi - mini

    # mlab.contour3d(x_mesh, y_mesh, z_mesh, g_mesh)
    # mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh))
    mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh), vmin = .05 * maxi, vmax = .5 * maxi)

    mlab.show()


def test_anim():
    x_coord = np.array([0.0, 1.0, 0.0, -1.0])
    y_coord = np.array([1.0, 0.0, -1.0, 0.0])
    z_coord = np.array([0.2, -0.2, 0.2, -0.2])

    plt = mlab.points3d(x_coord, y_coord, z_coord)

    msplt = plt.mlab_source

    @mlab.animate(delay = 100)
    def anim():
        angle = 0.0
        while True:
            x_coord = np.array([np.sin(angle), np.cos(angle), -np.sin(angle), -np.cos(angle)])
            y_coord = np.array([np.cos(angle), -np.sin(angle), -np.cos(angle), np.sin(angle)])
            msplt.set(x = x_coord, y = y_coord)
            yield
            angle += 0.1

    anim()
    mlab.show()


def test_hyd_anim():
    @mlab.animate(delay = 100)
    def anim(sim):
        angle = 0.0
        while True:
            x_coord = np.array([np.sin(angle), np.cos(angle), -np.sin(angle), -np.cos(angle)])
            y_coord = np.array([np.cos(angle), -np.sin(angle), -np.cos(angle), np.sin(angle)])
            msplt.set(x = x_coord, y = y_coord)
            yield
            angle += 0.1

    sim = ion.SphericalHarmonicSpecification(
        'test',
        r_bound = 50 * bohr_radius,
        r_points = 400,
        l_bound = 200,
        theta_points = 360,
        initial_state = ion.HydrogenBoundState(1, 0),
        electric_potential = ion.Rectangle(20 * asec, 80 * asec, 1 * atomic_electric_field),
        time_final = 100 * asec,
    ).to_simulation()

    sim.run_simulation(progress_bar = True, callback = anim)

    r, theta = sim.mesh.r, sim.mesh.theta

    g_r_theta = np.abs(sim.mesh.space_g)

    g_func = interp.RectBivariateSpline(r, theta, g_r_theta)

    def g_func_func(x_mesh, y_mesh, z_mesh):
        r_mesh = np.sqrt(x_mesh ** 2 + y_mesh ** 2 + z_mesh ** 2)
        print(si.utils.bytes_to_str(r_mesh.nbytes))
        theta_mesh = np.arccos(z_mesh / r_mesh)
        print(si.utils.bytes_to_str(theta_mesh.nbytes))
        return g_func.ev(r_mesh, theta_mesh)

    # x_max = sim.spec.r_bound
    x_max = 20 * bohr_radius
    x = np.linspace(-x_max, x_max, 100)
    x_mesh, y_mesh, z_mesh = np.meshgrid(x, x, x, indexing = 'ij')

    g_mesh = g_func_func(x_mesh, y_mesh, z_mesh)
    print(si.utils.bytes_to_str(g_mesh.nbytes))

    mini = g_mesh.min()
    maxi = g_mesh.max()
    # rng = maxi - mini

    # mlab.contour3d(x_mesh, y_mesh, z_mesh, g_mesh)
    # mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh))
    plt = mlab.pipeline.volume(mlab.pipeline.scalar_field(g_mesh), vmin = .05 * maxi, vmax = .5 * maxi)
    msplt = plt.mlab_source

    mlab.show()


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        test_anim()
        # while True:
        #     try:
        #         pass
        #         # test_rect()
        #         # test_sph()
        #         # test_hyd()
        #     except traits.trait_errors.TraitError:
        #         pass

# sim = ion.SphericalHarmonicSpecification(
#     'test',
#     r_bound = 20 * bohr_radius,
#     r_points = 100,
#     l_bound = 20,
#     theta_points = 100,
#     initial_state = ion.HydrogenBoundState(1, 0),
# ).to_simulation()
#
# # r = sim.mesh.r
# r = np.linspace(0, 10, 100)
# theta = sim.mesh.theta
# phi = np.linspace(0, twopi, 100)
#
# r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, phi, indexing = 'ij')
#
# # r_mesh = sim.mesh.r_theta_mesh
# # theta_mesh = sim.mesh.theta_mesh
# # phi_mesh = sim.mesh
#
# g_r_theta = sim.mesh.space_g.real
# # g_mesh = np.repeat(g_r_theta, 100).reshape(r_mesh.shape)
# g_mesh = 1 / (r_mesh + 1)
#
# print(si.utils.bytes_to_str(g_mesh.nbytes))
#
# x_mesh = r_mesh * np.sin(theta_mesh) * np.cos(phi_mesh)
# y_mesh = r_mesh * np.sin(theta_mesh) * np.sin(phi_mesh)
# z_mesh = r_mesh * np.cos(theta_mesh)
#
# print(x_mesh.shape)
# print(y_mesh.shape)
# print(z_mesh.shape)
# print(g_mesh.shape)
#
# print(x_mesh)
# print(y_mesh)
# print(z_mesh)
# print(g_mesh)
# x = np.linspace(-5, 5, 100)
# x_mesh, y_mesh, z_mesh = np.meshgrid(x, x, x)
