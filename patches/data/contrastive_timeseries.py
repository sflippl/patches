"""Contrastive timeseries.
"""

import numpy as np
import torch
import torch.utils.data as data

__all__ = ['Contrastive1DTimeSeries']

class Contrastive1DTimeSeries(data.Dataset):
    """Creates a contrastive sample of a one-dimensional time series.

    The dimensionality here refers to the shape. This means that every sample
    returned consists of a numpy array with dimensions sample x time x variable.
    """

    def __init__(self, data, n_contrasts=10, timesteps=range(1, 6)):
        self.data = data
        self.n_contrasts = n_contrasts
        self.timesteps=timesteps

    @property
    def timepoints(self):
        return self.data.shape[0]

    @property
    def n_vars(self):
        return self.data.shape[1]

    @property
    def n_timesteps(self):
        return len(self.timesteps)

    def __len__(self):
        return self.timepoints - max(self.timesteps)

    def __getitem__(self, idx):
        current_value = self.data[idx, :].reshape(1, 1, self.n_vars)
        future_values = self.data[[idx+t for t in self.timesteps], :].\
                             reshape(1, self.n_timesteps, self.n_vars)
        contrastive_idx = torch.randint(0, self.timepoints,
                                        (self.n_contrasts, ))
        contrastive_values = np.concatenate([
            self.data[[c_idx]*self.n_timesteps, :].\
                 reshape(1, self.n_timesteps, self.n_vars)\
            for c_idx in contrastive_idx
        ])
        return {
            'current_values': _numpy_to_torch(current_value),
            'future_values': _numpy_to_torch(future_values),
            'contrastive_values': _numpy_to_torch(contrastive_values)
        }

def _numpy_to_torch(x):
    return torch.from_numpy(x.astype(np.float32))
