"""Test patches.datasets.pilgrimm.layers.

There are going to be some random statements in these tests. By specifying seeds and
using extremely likely scenarios, these tests are likely always appropriate, even for
future modifications. Just in case, it might be necessary to change the seed though.
"""

import unittest

import numpy as np

from .. import messages as msg
from .. import layers

class TestShapeLayerInitialization(unittest.TestCase):
    """Test if the initialization works correctly.
    """

    def setUp(self):
        self.std_shapes = [np.array([[1]]), np.array([[-1]])]
        self.std_transition = np.array([[1., 0], [0, 1.]])
        self.std_initial = np.array([0.5, 0.5])

    def test_transition_probabilities(self):
        """Are the transition probabilities tested?
        """
        with self.assertRaises(AttributeError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=1)
        with self.assertRaises(ValueError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=np.array([[0.95], [0.95]]))

    def test_initial_distribution(self):
        """Is the initial distribution tested?
        """
        with self.assertRaises(AttributeError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=self.std_transition,
                              initial_distribution=1)
        with self.assertRaises(ValueError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=self.std_transition,
                              initial_distribution=np.array([0.1, 0.1, 0.8]))
        with self.assertRaises(ValueError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=self.std_transition,
                              initial_distribution=np.array([0.1, 0.1]))
        with self.assertRaises(ValueError):
            layers.ShapeLayer(width=10,
                              shapes=self.std_shapes,
                              transition_probabilities=self.std_transition,
                              initial_distribution=np.array([-0.5, 1.5]))

class TestShapeLayer(TestShapeLayerInitialization):
    """Test the shape layer's behavior.
    """

    def setUp(self):
        super().setUp()
        self.det_layer = layers.ShapeLayer(width=10,
                                           shapes=self.std_shapes,
                                           transition_probabilities=self.std_transition,
                                           initial_distribution=np.array([1, 0]))
        self.magic_layer = layers.ShapeLayer(width=10,
                                             shapes=self.std_shapes,
                                             transition_probabilities=self.std_transition,
                                             magic=True,
                                             initial_distribution=np.array([1, 0]))
        self.big_shapes = layers.ShapeLayer(width=10,
                                            shapes=[np.array([[1]*10]), np.array([[-1]*10])],
                                            transition_probabilities=self.std_transition,
                                            initial_distribution=np.array([1, 0]))
        self.rs = np.random.RandomState(seed=100)
        self.msg_stack = msg.MessageStack()

    def test_available_positions(self):
        """Are the available positions correct?
        """
        det_av_pos = self.det_layer._available_positions(self.det_layer.shapes[0])
        self.assertEqual(det_av_pos, {'x': 10, 'y': 10})
        big_av_pos = self.big_shapes._available_positions(self.big_shapes.shapes[0])
        self.assertEqual(big_av_pos, {'x': 1, 'y': 10})

    def test_get_array(self):
        """Does the 'get_array' method work?
        """
        arr = self.det_layer.get_array([0, 5], 0)
        self.assertTrue(np.all(np.logical_or(np.isnan(arr), arr==1.)))

    def test_normal_samples(self):
        """Can we sample normally?
        """
        self.det_layer.last_shape = 0
        self.det_layer.sample(self.rs, self.msg_stack)
        self.assertEqual(self.det_layer.shape, np.array([0]))
        self.assertTrue(np.all(self.det_layer.latent_array()==0))
        print(self.det_layer.to_array())
        det_array = self.det_layer.to_array()
        self.assertTrue(np.all(np.logical_or(np.isnan(det_array), det_array==1)))
        self.det_layer.forget()
        self.assertIsNone(self.det_layer.last_shape)

    def test_is_in(self):
        """Does the 'is_in' method work?
        """
        self.det_layer.last_shape = 0
        self.det_layer.sample(self.rs, self.msg_stack)
        old_array = self.det_layer.get_array(self.det_layer.location[-1],
                                             self.det_layer.shape[-1, 0])
        self.assertTrue(self.det_layer.is_in(old_array))
        self.assertFalse(self.det_layer.is_in(np.full(fill_value=np.nan, shape=(10, 10))))
        self.assertTrue(self.det_layer.is_in(np.full(fill_value=1., shape=(10, 10))))
        different_array = self.det_layer.get_array([9-self.det_layer.location[-1][0],
                                                    9-self.det_layer.location[-1][0]],
                                                   0)
        self.assertFalse(self.det_layer.is_in(different_array))

    def test_magic(self):
        """Test if the message stack accumulates.
        """
        self.magic_layer.sample(self.rs, self.msg_stack)
        self.assertIsInstance(self.msg_stack[0], msg.ForgetMessage)
        self.assertIsNotNone(self.magic_layer.last_shape)
        self.msg_stack(self.magic_layer)
        self.assertIsNone(self.magic_layer.last_shape)

class TestOccluder(TestShapeLayer):
    """Test the occluding layer's behavior.
    """

    def setUp(self):
        super().setUp()
        self.det_occ = layers.uniform_occlusion(occlusion_prob=1)
        self.magic_occ = layers.uniform_occlusion(occlusion_prob=1, magic=True)
        self.no_occ = layers.uniform_occlusion(occlusion_prob=0, magic=True)
        self.rs = np.random.RandomState(seed=100)
        self.msg_stack = msg.MessageStack()

    def occlusions(self, layer):
        """Test occlusions.
        """
        layer.sample(self.rs, self.msg_stack)
        self.assertEqual(list(layer.latent_array().shape), [1, 0])

    def test_normal_occlusion(self):
        """Do normal occlusions work?
        """
        self.occlusions(self.det_occ)
        self.assertEqual(len(self.msg_stack), 0)

    def test_magic_occlusions(self):
        """Do magical occlusions work?"""
        self.occlusions(self.magic_occ)
        self.assertTrue(np.all(self.magic_occ.to_array()==0))
        self.assertIsInstance(self.msg_stack[0], msg.ForgetMessage)
        self.det_layer.last_shape = 1
        self.det_layer.sample(self.rs, msg.MessageStack())
        self.msg_stack(self.det_layer)
        self.assertIsNone(self.det_layer.last_shape)
        self.msg_stack.restart()
        self.occlusions(self.no_occ)
        self.assertTrue(np.all(np.isnan(self.no_occ.to_array())))
        self.det_layer.last_shape = 1
        self.det_layer.sample(self.rs, msg.MessageStack())
        self.msg_stack(self.det_layer)
        self.assertIsNotNone(self.det_layer.last_shape)
