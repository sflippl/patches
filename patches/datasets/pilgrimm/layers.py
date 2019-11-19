"""Defines layers in the pilgrim model.
"""

import abc

import numpy as np

from .messages import ForgetMessage

class Layer(abc.ABC):
    """A layer of a pilgrimm atom.

    A layer must have a sample, a to_array, a latents, and
    a forget function. By default,
    the forget function does nothing, but the remaining functions
    must be implemented in any inherited class.
    """

    def __init__(self, width, magic=False):
        self.width = width
        self.magic = magic
        self.last_id = None

    @abc.abstractmethod
    def sample(self, random_state):
        """Draw a sample from the layer. You must pass a random state.
        """

    @abc.abstractmethod
    def to_array(self):
        """Returns the array of the layer. np.nan implies that the
        position is not filled.
        """

    @abc.abstractmethod
    def latent_array(self):
        """Returns the latent array. Can be empty, but must have the right
        sample dimensionality.
        """

    def forget(self):
        pass

    @abc.abstractmethod
    def is_in(self, array):
        """Checks if the last value is inside a certain array."""

    def magical(self, array, message_stack):
        if self.last_id is not None:
            message_stack.pop(self.last_id)
        def cond(layer):
            return layer.is_in(array)
        self.last_id = message_stack.add_message(ForgetMessage(cond=cond))

class ShapeLayer(Layer):
    """A layer of a pilgrimm atom.

    One layer consists of a set of possible states and transition
    probabilities between them. These states can be hierarchically
    ordered, which makes assignment with this class directly a bit
    complicated. For this reason, the access functions are often
    recommended.
    """

    def __init__(self, width, shapes, transition_probabilities,
                 magic=False,
                 initial_distribution=None):
        super().__init__(width, magic)
        self._validate_init(shapes, transition_probabilities,
                            initial_distribution)
        self.width = width
        self.shape = np.empty(shape=(0, 1), dtype=int)
        self.location = np.empty(shape=(0, 2), dtype=int)
        self.last_shape = None

    def _validate_init(self, shapes, transition_probabilities,
                       initial_distribution=None):
        self.shapes = shapes
        self.n_shapes = len(shapes)
        self.transition_probabilities = transition_probabilities
        self.initial_distribution = initial_distribution
        if initial_distribution is None:
            self.initial_distribution = np.array([1/self.n_shapes]*self.n_shapes,
                                                 dtype=np.float32)
        trans_shape = self.transition_probabilities.shape
        if list(trans_shape) != [
            self.n_shapes, self.n_shapes
        ]:
            raise ValueError('Transition probabilities have shape '
                             '{}, but should have shape {}.'.\
                             format(list(trans_shape), [self.n_shapes,
                                                        self.n_shapes]))
        init_shape = self.initial_distribution.shape
        if list(init_shape) != [self.n_shapes]:
            raise ValueError('Initial distribution has shape '
                             '{}, but should have shape {}.'.\
                             format(list(init_shape), [self.n_shapes]))
        if abs(sum(self.initial_distribution) - 1.0) > 1e-5:
            raise ValueError('Initial distribution does not sum up '
                             'correctly.')
        if any(self.initial_distribution < 0):
            raise ValueError('Probabilities must be non-negative.')

    def sample(self, random_state, message_stack):
        """Samples from a layer while saving relevant information
        to the message passer.
        """
        if self.last_shape is None:
            new_shape_idx = random_state.choice(
                range(self.n_shapes),
                p=self.initial_distribution
            )
        else:
            new_shape_idx = random_state.choice(
                range(self.n_shapes),
                p=self.transition_probabilities[self.last_shape]
            )
        self.last_shape = new_shape_idx
        self.shape = np.append(self.shape, [[new_shape_idx]], axis=0)
        _av_locs = self._available_positions(self.shapes[new_shape_idx])
        new_locs = [random_state.randint(0, _av_locs['y'], 1)[0], 
                    random_state.randint(0, _av_locs['x'], 1)[0]]
        self.location = np.append(self.location,
                                   [new_locs],
                                   axis=0)
        if self.magic:
            new_array = self.get_array(new_locs, new_shape_idx)
            self.magical(new_array, message_stack)

    def get_array(self, locs, shape_idx):
        arr = np.full(shape=(self.width, self.width), fill_value=np.nan)
        new_shape = self.shapes[shape_idx]
        arr[locs[0]:(locs[0]+new_shape.shape[0]),
            locs[1]:(locs[1]+new_shape.shape[1])] = new_shape
        return arr

    def _available_positions(self, shape):
        return {
            'x': self.width-shape.shape[1]+1,
            'y': self.width-shape.shape[0]+1
        }

    def forget(self, cond=None):
        if (cond is None) or (cond(self)):
            self.last_value = None

    def to_array(self):
        lst_arrs = [np.expand_dims(self.get_array(location, shape[0]), axis=0)
                    for location, shape in zip(self.location, self.shape)]
        arr = np.concatenate(lst_arrs, axis=0)
        return arr

    def latent_array(self):
        return self.shape

    def is_in(self, array):
        last_array = self.get_array(self.location[-1], self.shape[-1,0])
        return np.all(np.logical_or(np.isnan(last_array), np.logical_not(np.isnan(array))))

class OccludingLayer(Layer):
    """Creates a random occlusion with a certain probability at every timepoint.

    Random occlusions are defined by any function returning an array. If 'magic'
    is true, any object that completely vanishes behind an occlusion will forget
    its history.

    Args:
        width: What is the panel's width?
        occlusion_fun: The function returning the occlusion. It should take in
            two arguments, a np.random.RandomState, and the layer's width. The
            function is expected to return a two-dimensional array.
        magic: If an occlusion covers an object, will that object be forgotten?
    """

    def __init__(self, width, occlusion_fun, magic=False):
        super().__init__(width, magic)
        self.width = width
        self.occlusion_fun = occlusion_fun
        self.magic = magic
        self.array = np.empty(shape=(0, width, width), dtype=np.float32)
        self.last_id = None

    def sample(self, random_state, message_stack):
        new_array = self.occlusion_fun(random_state, self.width)
        self.array = np.append(
            self.array,
        )
        if self.magic:
            self.magical(new_array, message_stack)

    def is_in(self, array):
        return False

    def latent_array(self, array):
        return np.empty(shape=(self.array.shape[0], 0), dtype=np.float32)