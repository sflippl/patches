"""Pilgrimm: A positionally invariant layered geometric Markov model.
"""

import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import plotnine as gg

from .messages import MessageStack
import patches.utils as utils

class Atom:
    """An atom of the pilgrim model.
    """

    def __init__(self, layers, width=None):
        self.width = width or layers[0].width
        for layer in layers:
            if layer.width != self.width:
                raise ValueError('Widths and heights of all layers '
                                 'must be the same.')
        self.layers = layers
        self.n_samples = 0
        self.message_stack = MessageStack()

    def sample(self, random_state):
        for layer in self.layers:
            self.message_stack(layer)
            layer.sample(random_state=random_state,
                         message_stack=self.message_stack)
        self.message_stack.restart()
        self.n_samples += 1

    def to_array(self):
        arr = np.concatenate([
            np.expand_dims(layer.to_array(), -1)\
            for layer in self.layers
        ], axis=-1)
        return arr

    def flat_array(self):
        arr = np.full(shape=(self.n_samples, self.width, self.width), fill_value=np.nan)
        for layer in self.layers:
            arr = np.where(np.isnan(arr), layer.to_array(), arr)
        np.nan_to_num(arr, copy=False)
        return arr

    def latent_array(self):
        arr = np.concatenate([
            layer.latent_array() for layer in self.layers
        ], axis=-1)
        return arr

class Pilgrimm:
    """A positionally invariant layered geometric Markov model.

    This class can best be understood as a generalization of the
    compositional binary model, the occlusion model, and the geometric
    model. It consists of several layers and several regions of interest.
    The model has a number of parameters that probe different aspects
    of predictability and object permanence. It works in discrete time
    and its position is completely random.
    """

    def __init__(self, atoms, samples=0, seed=None, random_state=None):
        for atom in atoms:
            if not isinstance(atom, Atom):
                raise ValueError('Atoms must be of type Atom, '
                                 'but one atom is of type {}'.\
                                 format(type(atom)))
        self.atoms = atoms
        if random_state is None:
            self.random_state = np.random.RandomState(seed=seed)
        else:
            self.random_state = random_state
        self.n_samples = 0
        self.sample(samples)

    def sample(self, samples):
        """This function samples from the pilgrimm model.

        samples: The number of samples you wish to draw.
        """
        for atom in self.atoms:
            for __ in range(samples):
                atom.sample(random_state=self.random_state)
        self.n_samples += samples

    def to_array(self):
        """This function returns the resulting array in a two-dimensional
        format mapping (samples)x(width)x(width)x(height)x(atoms).
        """
        arr = np.concatenate([
            np.expand_dims(atom.to_array(), -1) for atom in self.atoms
        ], axis=-1)
        return arr

    def flat_array(self, format=None):
        """This function returns the resulting array from a two-dimensional
        perspective, in the format (samples)x(width)x(width)x(atoms).

        Args:
            format: How should the worlds be split up between width and height?
                Default adds an extra dimension."""
        if format is None:
            arr = np.concatenate([
                np.expand_dims(atom.flat_array(), -1) for atom in self.atoms
            ], axis=-1)
            return arr
        if format[0]*format[1] != len(self.atoms):
            raise ValueError('The format is not compatible with the number of atoms.')
        arr = np.concatenate([
            np.concatenate([
                atom.flat_array() for atom in self.atoms[(i*format[1]):((i+1)*format[1])]
            ], axis=2) for i in range(format[0])
        ], axis=1)
        return arr

    def latent_array(self):
        arr = np.concatenate([atom.latent_array() for atom in self.atoms], axis=1)
        return arr

    def animate(self, n=None, file=None, format=None, html=False):
        format = format or (1, len(self.atoms))
        tmpdir = tempfile.gettempdir()
        _file = file or '{}/ipython_animation.gif'.format(tmpdir)
        arr = self.flat_array(format)
        fig = plt.figure()
        plt.axis('off')
        if n is None:
            n = arr.shape[0]
        def updatefig(i, *args):
            im = plt.imshow(arr[i], animated=True, norm=mcolors.Normalize(vmin = arr.min(), vmax=arr.max()))
            return im, 
        ani = animation.FuncAnimation(fig, updatefig, frames=range(n), blit=True,
                                      repeat=True)
        if html:
            try:
                from IPython.display import HTML
                vid = HTML(ani.to_html5_video())
                plt.close()
                return vid
            except NameError as e:
                raise NotImplementedError('You can only call this function without a filepath from '
                                          'an IPython environment.')\
                    from e
        ani.save(_file, writer='pillow')
        plt.close()
        if file is None:
            try:
                from IPython.display import Image
                return Image(filename=_file)
            except NameError as e:
                raise NotImplementedError('You can only call this function without a filepath from '
                                          'an IPython environment.')\
                    from e

    def latent_evolution(self, **kwargs):
        df = utils.array_to_dataframe(self.latent_array())\
                  .rename(columns = {'dim0': 'time',
                                     'dim1': 'latent_dimension',
                                     'array': 'value'})
        img = (gg.ggplot(df, gg.aes(x='time', y='value')) +
                  gg.geom_line() +
                  gg.facet_wrap('latent_dimension', dir='v', **kwargs) +
                  gg.labs(x='Time', y='', title='Latent values'))
        return img
