"""Test utils.dataframe.
"""

import unittest

import numpy as np

from .. import dataframe as df

class TestArrayToDataFrame(unittest.TestCase):
    """Test array_to_dataframe function.
    """

    def setUp(self):
        """Sets up the two convertable arrays.
        """
        self.dat1 = np.empty(shape=(0, ))
        self.dat2 = np.array([[1, 2]])

    def test_fail(self):
        """Test if the function fails appropriately.
        """
        with self.assertRaises(TypeError):
            df.array_to_dataframe(1)

    def test_correct_output(self):
        """Test the output dataframe.
        """
        dat1 = df.array_to_dataframe(self.dat1)
        self.assertEqual(len(dat1), 0)
        self.assertEqual(list(dat1.columns), ['dim0', 'array'])
        dat2 = df.array_to_dataframe(self.dat2)
        self.assertEqual(len(dat2), 2)
        self.assertEqual(len(dat2.columns), 3)
