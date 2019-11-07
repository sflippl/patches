"""Provides latent and input layer in a hidden Markov model.
"""

import numpy as np
import torch
import torch.utils.data as data

__all__ = ['HiddenMarkovModel', 'Hidden2DMarkovModel']

class HiddenMarkovModel(data.Dataset):
    """The Hidden Markov Model consists of an input time series (time x var) and
    a latent time series (time x var)."""
    def __init__(self, data, latents, timesteps=range(1, 6)):
        super().__init__()
        self.data = data
        self.latents = latents
        self.timesteps = timesteps

    def __len__(self):
        return self.timepoints - max(self.timesteps)

    @property
    def timepoints(self):
        return self.data.shape[0]

    @property
    def n_vars(self):
        return self.data.shape[1]

    @property
    def n_timesteps(self):
        return len(self.timesteps)

    def __getitem__(self, idx):
        return {
            'input': _numpy_to_torch(self.data[idx, :]),
            'latent_values': _numpy_to_torch(self.latents[idx, :]),
            'future_latent_values': _numpy_to_torch(
                self.latents[
                    [idx+t for t in self.timesteps], :
                ]
            )
        }

class Hidden2DMarkovModel(data.Dataset):
    """The Hidden Markov Model consists of an input time series (time x var x var) and
    a latent time series (time x var)."""
    def __init__(self, data, latents, timesteps=range(1, 6)):
        super().__init__()
        self.data = data
        self.latents = latents
        self.timesteps = timesteps

    def __len__(self):
        return self.timepoints - max(self.timesteps)

    @property
    def timepoints(self):
        return self.data.shape[0]

    @property
    def n_vars(self):
        return (self.data.shape[1], self.data.shape[2])

    @property
    def n_timesteps(self):
        return len(self.timesteps)

    def __getitem__(self, idx):
        return {
            'input': _numpy_to_torch(self.data[idx, :]),
            'latent_values': _numpy_to_torch(self.latents[idx, :]),
            'future_latent_values': _numpy_to_torch(
                self.latents[
                    [idx+t for t in self.timesteps], :
                ]
            )
        }

def _numpy_to_torch(x):
    return torch.from_numpy(x.astype(np.float32))
