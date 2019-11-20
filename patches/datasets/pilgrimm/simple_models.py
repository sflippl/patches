"""Simple models include the compositional binary model, its geometric variant,
and its occluded variant."""

import numpy as np

from . import layers
from . import pilgrimm
from . import shapes as shps

def _change_matrix(change_prob):
    return np.array([[1-change_prob, change_prob],
                     [change_prob, 1-change_prob]])

def compositional_binary_model(width=None, change_probs=None, samples=0,
                               seed=None, random_state=None):
    shapes = [np.array([[1.]]), np.array([[-1.]])]
    cbm = compositional_geometric_model(width=width, change_probs=change_probs,
                                        shapes=shapes, samples=samples, seed=seed,
                                        random_state=random_state)
    return cbm

def compositional_geometric_model(width=10, change_probs=None, shapes=None,
                                  samples=0, seed=None, random_state=None):
    shapes = shapes or [np.array([[1., 1.]]), np.array([[1.], [1.]])]
    change_probs = change_probs or [0.05]
    if not isinstance(shapes[0], list):
        shapes = [shapes]*len(change_probs)
    atoms = [
        pilgrimm.Atom([
            layers.ShapeLayer(
                width,
                shapes=_shapes,
                transition_probabilities=_change_matrix(change_prob)
            )
        ]) for _shapes, change_prob in zip(shapes, change_probs)
    ]
    cgm = pilgrimm.Pilgrimm(atoms, samples=samples, random_state=random_state,
                            seed=seed)
    return cgm

def occluded_binary_model(width=10, change_probs=None, occlusion_probs=None,
                          occlusion_range=0, magic=False,
                          samples=0, seed=None, random_state=None):
    shapes = [np.array([[1.]]), np.array([[-1.]])]
    obm = occluded_geometric_model(width=width, change_probs=change_probs,
                                   occlusion_probs=occlusion_probs,
                                   occlusion_range=occlusion_range, magic=magic,
                                   shapes=shapes, samples=samples, seed=seed,
                                   random_state=random_state)
    return obm

def occluded_geometric_model(width=10, change_probs=None, occlusion_probs=0.1,
                             shapes=None, occlusion_range=0, magic=False,
                             samples=0, seed=None, random_state=None):
    shapes = shapes or [np.array([[1., 1.]]), np.array([[1.], [1.]])]
    change_probs = change_probs or [0.05]
    if not isinstance(shapes[0], list):
        shapes = [shapes]*len(change_probs)
    if not isinstance(occlusion_probs, list):
        occlusion_probs = [occlusion_probs]*len(change_probs)
    if not isinstance(magic, list):
        magic = [magic]*len(change_probs)
    if len(change_probs) != len(occlusion_probs):
        raise ValueError('Change_probs and occlusion_probs must have the same '
                         'length.')
    atoms = [
        pilgrimm.Atom([
            layers.uniform_occlusion(width, occlusion_range, magic=_magic,
                                     occlusion_prob=occlusion_prob),
            layers.ShapeLayer(width, shapes=_shapes,
                              transition_probabilities=_change_matrix(change_prob))
        ]) for _shapes, change_prob, occlusion_prob, _magic\
            in zip(shapes, change_probs, occlusion_probs, magic)
    ]
    ogm = pilgrimm.Pilgrimm(atoms, samples=samples, random_state=random_state,
                            seed=seed)
    return ogm

def _seq_matrix(n):
    return np.identity(n)[list(range(1,n))+[0]]

def sequential_geometric_model(width=10, occlusion_probs=None, shapes=None,
                               occlusion_range=0, magic=False, samples=0,
                               seed=None, random_state=None):
    shapes = shapes or [np.array([[1., 1.]]), np.array([[1.], [1.]])]
    occlusion_probs = occlusion_probs or [0.1]
    if not isinstance(shapes[0], list):
        shapes = [shapes]*len(occlusion_probs)
    if not isinstance(magic, list):
        magic = [magic]*len(occlusion_probs)
    if len(magic) != len(occlusion_probs):
        raise ValueError('Magic and occlusion_probs must have the same length.')
    atoms = [
        pilgrimm.Atom([
            layers.uniform_occlusion(width, occlusion_range, magic=_magic,
                                     occlusion_prob=occlusion_prob),
            layers.ShapeLayer(width, shapes=_shapes,
                              transition_probabilities=_seq_matrix(len(_shapes)))
        ]) for _shapes, occlusion_prob, _magic\
            in zip(shapes, occlusion_probs, magic)
    ]
    sgm = pilgrimm.Pilgrimm(atoms, samples=samples, random_state=random_state,
                            seed=seed)
    return sgm

def layered_geometric_model(width=10, shapes=None, change_probs=None,
                            samples=0, seed=None,
                            random_state=None):
    shapes = shapes or [[shps.horizontal(2), shps.vertical(2)],
                        [shps.diagonal(2), shps.antidiagonal(2)]]
    change_probs = change_probs or [0.05, 0.5]
    layrs = [
        layers.ShapeLayer(width, shapes=_shapes,
                          transition_probabilities=_change_matrix(change_prob))\
        for _shapes, change_prob in zip(shapes, change_probs)
    ]
    lgm = pilgrimm.Pilgrimm(
        [pilgrimm.Atom(layrs)], samples=samples, random_state=random_state, seed=seed
    )
    return lgm

def object_slots_model(width=10, shapes=None, occlusion_ranges=0, occlusion_probs=0.1,
                       appearance_probs=0.01,
                       magic=False, samples=0, random_state=None, seed=None):
    shapes = shapes or [np.array([[1., 1.]]), np.array([[1.], [1.]])]
    if not isinstance(appearance_probs, list):
        appearance_probs = [appearance_probs]*len(shapes)
    if not isinstance(occlusion_probs, list):
        occlusion_probs = [occlusion_probs]*len(shapes)
    if not isinstance(occlusion_ranges, list):
        occlusion_ranges = [occlusion_ranges]*len(shapes)
    if not isinstance(magic, list):
        magic = [magic]*len(shapes)
    inp_shapes = [[np.empty(shape=(0, 0)), shape] for shape in shapes]
    atoms = [
        pilgrimm.Atom([
            layers.uniform_occlusion(width, occlusion_range, magic=_magic,
                                     occlusion_prob=occlusion_prob),
            layers.ShapeLayer(width, shapes=_shapes,
                              transition_probabilities=np.array([
                                  [1-appearance_prob, appearance_prob],
                                  [0., 1.]
                              ]),
                              initial_distribution=np.array([1., 0.]))
        ]) for occlusion_range, _shapes, occlusion_prob, _magic, appearance_prob\
            in zip(occlusion_ranges, inp_shapes, occlusion_probs, magic,
                   appearance_probs)
    ]
    osm = pilgrimm.Pilgrimm(atoms, samples=samples, random_state=random_state,
                            seed=seed)
    return osm
