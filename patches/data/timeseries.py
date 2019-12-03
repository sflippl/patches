"""Returns a timeseries with input and target.
"""

import torch
import torch.utils.data as data
import numpy as np

class Timeseries(data.Dataset):
    """Provide a dataset with samples x timesteps x (remaining variables).
    """

    def __init__(self, data, timesteps=[0], device=None):
        self.data = data
        self.timesteps = timesteps
        time_range = max(max(timesteps), 0) - min(min(timesteps), 0)
        self.min_time = min(min(timesteps), 0)
        self.items_per_sample = self.data.shape[1]-(time_range+1)
        self.device = device

    def __len__(self):
        return self.data.shape[0]*self.items_per_sample

    def to(self, device=None):
        self.device = device

    def __getitem__(self, idx):
        idx_0 = int(np.floor(idx/self.items_per_sample))
        idx_1 = int(idx % self.items_per_sample)
        input = self.data[idx_0, [idx_1-self.min_time]]
        target_range = [idx_1+t-self.min_time for t in self.timesteps]
        target = self.data[idx_0, target_range]
        input = torch.from_numpy(input.astype(np.float32))\
                     .to(device=self.device)
        target = torch.from_numpy(target.astype(np.float32))\
                      .to(device=self.device)
        return {'input': input, 'target': target}
