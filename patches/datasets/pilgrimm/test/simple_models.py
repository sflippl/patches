"""Test patches.datasets.pilgrimm.simple_models.
"""

import unittest

import numpy as np

from .. import simple_models as models

class TestCGM(unittest.TestCase):
    """Test the compositional geometric model.
    """

    def test_cgm(self):
        """Test the CGM."""
        cgm = models.compositional_geometric_model(width=10, change_probs=[0, 1], samples=5, seed=100)
        lat_array = cgm.latent_array()
        self.assertEqual(len(np.unique(lat_array[:, 0])), 1)
        self.assertEqual(len(np.unique(lat_array[:, 1])), 2)

    def test_cbm(self):
        """Test the CBM."""
        cbm = models.compositional_binary_model(width=1, change_probs=[0, 1], samples=5, seed=100)
        self.assertTrue(np.all(cbm.flat_array()**2==1))

class TestOGM(unittest.TestCase):
    """Test the occluded geometric model.
    """

    def test_ogm(self):
        """Test the OGM."""
        ogm = models.occluded_geometric_model(width=10, change_probs=[0, 0], magic=[False, True],
                                              samples=20, seed=100, occlusion_probs=1)
        lat_array = ogm.latent_array()
        self.assertEqual(len(np.unique(lat_array[:, 0])), 1)
        self.assertEqual(len(np.unique(lat_array[:, 1])), 2)

    def test_obm(self):
        """Test the OBM."""
        obm = models.occluded_geometric_model(width=10, change_probs=[0, 0], magic=[False, True],
                                              samples=20, seed=100, occlusion_probs=1)
        self.assertTrue(np.all(obm.flat_array()==0))

class TestSGM(unittest.TestCase):
    """Test the sequential geometric model.
    """

    def test_sgm(self):
        """Test the SGM."""
        sgm = models.sequential_geometric_model(width=10, magic=[False, True], samples=20, seed=100,
                                                occlusion_probs=[1, 1])
        lat_array = sgm.latent_array()
        self.assertTrue(np.all((lat_array[1:, 0]+lat_array[:-1, 0])==1))
        self.assertFalse(np.all((lat_array[1:, 1]+lat_array[:-1, 1])==1))

class TestOSM(unittest.TestCase):
    """Test the object slots model.
    """

    def test_osm(self):
        """Test the OSM."""
        osm = models.object_slots_model(occlusion_probs=[0.2, 0.2], magic=[False, True], samples=20, seed=100,
                                        appearance_probs=0.5)
        lat_array = osm.latent_array()
        print(lat_array)
        diffs = (lat_array[1:] - lat_array[:-1])**2 != 0
        self.assertEqual(sum(diffs[:, 0]), 1)    
        self.assertGreater(sum(diffs[:, 1]), 1)
