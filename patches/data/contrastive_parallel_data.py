"""Contrastive parallel data.
"""

import numpy as np
import torch
import torch.utils.data as data

__all__ = ['Contrastive1DParallelData']

class Contrastive1DParallelData(data.Dataset):
    """Creates a contrastive sample of a 1D set of parallel data structures."""

    def __init__(self, input_data, parallel_data, n_contrasts=10, seed=None):
        super().__init__()
        self.input_data = input_data
        self.parallel_data = parallel_data
        self.n_contrasts = n_contrasts
        self.random_state = np.random.RandomState(seed=seed)

    @property
    def n_vars(self):
        return self.input_data.shape[1]

    def __len__(self):
        return self.input_data.shape[0]

    @property
    def n_parallels(self):
        return self.parallel_data.shape[1]

    def __getitem__(self, idx):
        input_values = self.input_data[idx, :].reshape(1, 1, self.n_vars)
        parallel_values = self.parallel_data[idx, :, :].reshape(1, self.n_parallels, self.n_vars)
        contrastive_idx = self.random_state.randint(0, len(self),
                                                    (self.n_contrasts, ))
        contrastive_values = self.parallel_data[contrastive_idx, :, :]
        return {
            'current_values': _numpy_to_torch(input_values),
            'future_values': _numpy_to_torch(parallel_values),
            'contrastive_values': _numpy_to_torch(contrastive_values)
        }

def _numpy_to_torch(x):
    return torch.from_numpy(x.astype(np.float32))
