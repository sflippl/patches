"""Provides latent and input layer in a hidden Markov model.
"""

import numpy as np
import torch
import torch.utils.data as data

__all__ = ['HiddenMarkovModel']

class HiddenMarkovModel(data.Dataset):
    """The hidden markov model consists of an input time series and a latent time series.

    Args:
        dataset: The expected format is (samples)x(timesteps)x(other variables).
        timesteps: How should the latent time series be moved with respect to the input
            timeseries? Default is 0.
    """

    def __init__(self, input_data, latent_data, timesteps=[0], device=None):
        self.input_data = input_data
        self.latent_data = latent_data
        self.timesteps = timesteps
        time_range = max(max(timesteps), 0) - min(min(timesteps), 0)
        self.min_time = min(min(timesteps), 0)
        self.items_per_sample = self.input_data.shape[1]-(time_range+1)
        self.device = device

    def __len__(self):
        return self.input_data.shape[0]*self.items_per_sample

    def to(self, device=None):
        self.device = device
        return self

    def __getitem__(self, idx):
        idx_0 = int(np.floor(idx/self.items_per_sample))
        idx_1 = int(idx % self.items_per_sample)
        input = self.input_data[idx_0, [idx_1-self.min_time]]
        target_range = [idx_1+t-self.min_time for t in self.timesteps]
        target = self.latent_data[idx_0, target_range]
        input = torch.from_numpy(input.astype(np.float32))\
                     #.to(device=self.device)
        target = torch.from_numpy(target.astype(np.float32))\
                      #.to(device=self.device)
        return {'input': input, 'target': target}
