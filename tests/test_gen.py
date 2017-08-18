import os
import unittest
import shutil

import numpy as np

import simulacra as si
import ionization as ion


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(THIS_DIR, 'temp-unit-testing')


class TestBeet(unittest.TestCase):
    def setUp(self):
        self.obj = si.Beet('foo')
        self.obj_name = 'foo'
        self.target_name = 'foo.beet'
        si.utils.ensure_dir_exists(TEST_DIR)

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_beet_names(self):
        self.assertEqual(self.obj.name, self.obj_name)
        self.assertEqual(self.obj.file_name, self.obj_name)

    def test_save_load(self):
        path = self.obj.save(target_dir = TEST_DIR)
        self.assertEqual(path, os.path.join(TEST_DIR, self.target_name))  # test if path was constructed correctly
        self.assertTrue(os.path.exists(path))  # path should actually exist on the system
        loaded = si.Beet.load(path)
        self.assertEqual(loaded, self.obj)  # beets should be equal, but NOT the same object
        self.assertEqual(loaded.uid, self.obj.uid)  # beets should have the same uid
        self.assertEqual(hash(loaded), hash(self.obj))  # beets should have the same hash
        self.assertIsNot(loaded, self.obj)  # beets should NOT be the same object


class TestSpecification(TestBeet):
    def setUp(self):
        self.obj = si.Specification('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        si.utils.ensure_dir_exists(TEST_DIR)


class TestSimulation(TestBeet):
    def setUp(self):
        self.obj = si.Simulation(si.Specification('baz'))
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        si.utils.ensure_dir_exists(TEST_DIR)


class TestElectricFieldSpecification:
    spec_type = ion.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('bar')
        self.obj_name = 'bar'
        self.target_name = 'bar.spec'
        si.utils.ensure_dir_exists(TEST_DIR)


class TestLineSpecification(TestElectricFieldSpecification, TestBeet):
    spec_type = ion.LineSpecification


class TestCylindricalSliceSpecification(TestElectricFieldSpecification, TestBeet):
    spec_type = ion.CylindricalSliceSpecification


class TestSphericalSliceSpecification(TestElectricFieldSpecification, TestBeet):
    spec_type = ion.SphericalSliceSpecification


class TestSphericalHarmonicSpecification(TestElectricFieldSpecification, TestBeet):
    spec_type = ion.SphericalHarmonicSpecification


class TestElectricFieldSimulation:
    spec_type = ion.ElectricFieldSpecification

    def setUp(self):
        self.obj = self.spec_type('baz').to_simulation()
        self.obj_name = 'baz'
        self.target_name = 'baz.sim'
        si.utils.ensure_dir_exists(TEST_DIR)

    def test_save_load__save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertEqual(loaded.mesh, pre_save_mesh)  # the new mesh should be equal
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh(self):
        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = False)

        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal
        self.assertIsNone(loaded.mesh)  # in fact, the loaded mesh shouldn't even exist

    def test_save_load__save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = True)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # even though we saved the mesh, we reinitialized, so it shouldn't have the same entries
        self.assertIsNot(loaded.mesh, pre_save_mesh)  # the new mesh will not be the same object, because it it gets swapped with a copy during the save

    def test_save_load__no_save_mesh__reinitialize(self):
        self.obj.mesh.g_mesh = np.ones(1000)  # replace g_mesh with dummy entry

        pre_save_mesh = self.obj.mesh.copy()

        path = self.obj.save(target_dir = TEST_DIR, save_mesh = False)
        loaded = ion.ElectricFieldSimulation.load(path, initialize_mesh = True)

        self.assertIsNotNone(loaded.mesh)
        self.assertNotEqual(loaded.mesh, pre_save_mesh)  # pre and post should not be equal

    def test_initial_norm(self):
        self.assertAlmostEqual(self.obj.mesh.norm(), 1)

    def test_initial_state_overlap(self):
        ip = self.obj.mesh.inner_product(self.obj.spec.initial_state)
        self.assertAlmostEqual(np.abs(ip) ** 2, 1)


class TestLineSimulation(TestElectricFieldSimulation, TestBeet):
    spec_type = ion.LineSpecification


class TestCylindricalSliceSimulation(TestElectricFieldSimulation, TestBeet):
    spec_type = ion.CylindricalSliceSpecification


class TestSphericalSliceSimulation(TestElectricFieldSimulation, TestBeet):
    spec_type = ion.SphericalSliceSpecification


class TestSphericalHarmonicSimulation(TestElectricFieldSimulation, TestBeet):
    spec_type = ion.SphericalHarmonicSpecification
