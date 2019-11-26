"""Test patches.datasets.pilgrimm.pilgrimm.
"""

import unittest

import numpy as np

from .. import messages as msg
from .. import layers
from .. import pilgrimm
from .layers import TestOccluder

class TestAtom(TestOccluder):
    """Test atoms.
    """

    def setUp(self):
        super().setUp()
        self.atom = pilgrimm.Atom([
            self.det_occ,
            self.det_layer
        ])
        self.magic_atom = pilgrimm.Atom([
            self.magic_occ,
            self.big_shapes
        ])
        self.no_atom = pilgrimm.Atom([
            self.no_occ,
            self.magic_layer
        ])
        self.rs = np.random.RandomState(seed=100)

    def test_atoms(self):
        self.atom.sample(self.rs)
        self.det_layer.last_shape = 1
        self.atom.sample(self.rs)
        self.assertEqual(self.det_layer.last_shape, 1)
        self.magic_atom.sample(self.rs)
        self.big_shapes.last_shape = 1
        self.magic_atom.sample(self.rs)
        self.assertEqual(self.big_shapes.last_shape, 0)
        self.assertEqual(len(self.magic_atom.message_stack), 0)
        self.no_atom.sample(self.rs)
        self.magic_layer.last_shape = 1
        self.no_atom.sample(self.rs)
        self.assertEqual(self.magic_layer.last_shape, 1)

class TestPilgrimm(TestAtom):
    """Test Pilgrimms.
    """

    def setUp(self):
        super().setUp()
        self.pilgrimm = pilgrimm.Pilgrimm([
            self.atom,
            self.magic_atom,
            self.no_atom
        ], seed=100)

    def test_pilgrimm(self):
        self.pilgrimm.sample(5)
        self.assertEqual(list(self.pilgrimm.flat_array().shape), [5, 10, 10, 3])
        self.assertEqual(list(self.pilgrimm.latent_array().shape), [5, 3])
