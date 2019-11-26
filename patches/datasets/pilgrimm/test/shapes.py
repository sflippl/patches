"""Tests patches.datasets.pilgrimm.shapes.
"""

import unittest

import numpy as np

from .. import shapes

class TestRectangles(unittest.TestCase):
    """Tests the rectangles.
    """

    def setUp(self):
        self.rects = [shapes.horizontal, shapes.vertical,
                      shapes.diagonal, shapes.antidiagonal]

    def test_fail(self):
        """Test if the appropriate input fails.
        """
        for rect in self.rects:
            with self.assertRaises(ValueError):
                rect(1)

    def test_rects(self):
        """Test if the returned input is indeed a numpy array.
        """
        for rect in self.rects:
            self.assertIsInstance(rect(3), np.ndarray)
