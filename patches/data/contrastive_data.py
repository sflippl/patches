"""This module provides ways to specify contrastive data.
"""

import numpy as np
import torch

class ContrastiveDataset(torch.utils.data.Dataset):
    """Implements a contrastive dataset.

    Args:
        data: The dataset must have the format
            (Samples)x(Prediction Channels)x(Remaining variables).
        contrast_size: How many contrasts should be provided?
        stride: How densely should the codes be sampled.
        contrast_type: How should the contrasts be determined? At the moment, the following
            options are implemented:
                - 'samples': The contrasts must come from another sample,
                - 'channels': The contrasts must come from the same sample, but other
                    prediction channels, or
                - 'both': The contrasts may both come from the same and a different sample.
        prediction_range: Integer. Which range should the prediction have?
            Default is no limit, which can be specified by 'all'. A prediction range is
            mostly useful for spatial or temporal data.
            If prediction range is 'all', within-sample contrasts are not accepted.
        **kwargs: Arguments from 'torch.utils.data.Dataloader' except for those arguments
            guiding the batch sampling, as one batch will consist of mutual contrasts.

    In particular, one batch will have the size contrast_size+1. One batch will have the form
    (Contrasts)x(Prediction Channels within range)x(Remaining variables).
    """

    def __init__(self, data, contrast_size=9, stride=1, contrast_type='samples',
                 prediction_range='all', seed=None,
                 **kwargs):
        super().__init__()
        self.prediction_range = prediction_range
        self.contrast_size = contrast_size
        self.contrast_type = contrast_type
        self.stride = stride
        self.random_state = np.random.RandomState(seed=seed)
        self.data = data
        if prediction_range!='all':
            if prediction_range <= 0:
                raise ValueError('Prediction range must be positive.')
            self.items_per_sample = (data.shape[1]-prediction_range)/stride
        else:
            self.items_per_sample = 1
        if (prediction_range == 'all') and (contrast_type != 'samples'):
            raise ValueError('Within-sample contrasts are not accepted for unlimited prediction '
                             'range.')

    def __getitem__(self, idx):
        idx_0 = int(np.floor(idx/self.items_per_sample))
        idx_1 = int(self.stride*(idx % self.items_per_sample))
        if self.contrast_type == 'samples':
            idx_0_range = np.arange(self.data.shape[0])
            idx_0_range = idx_0_range[idx_0_range != idx_0]
            idxs_0 = idx_0_range[
                self.random_state.randint(0, len(idx_0_range), size=self.contrast_size)
            ]
            idxs_0 = [idx_0] + list(idxs_0)
            if self.prediction_range == 'all':
                return torch.from_numpy(self.data[idxs_0].astype(np.float32))
            return torch.from_numpy(
                self.data[np.array(idxs_0), idx_1:(idx_1+self.prediction_range+1)].astype(np.float32)
            )
        data = [self.data[idx_0:(idx_0+1), idx_1:(idx_1+self.prediction_range+1)]]
        if self.contrast_type == 'channels':
            idxs_1 = self.random_state.randint(0, self.data.shape[1]-self.prediction_range,
                                               size=self.contrast_size)
            for i in idxs_1:
                data.append(self.data[[idx_0], i:(i+self.prediction_range+1)])
            return torch.from_numpy(np.concatenate(data, axis=0).astype(np.float32))
        idxs_0 = self.random_state.randint(0, self.data.shape[0], size=self.contrast_size)
        idxs_1 = self.random_state.randint(0, self.data.shape[1]-self.prediction_range,
                                           size=self.contrast_size)
        for i, j in zip(idxs_0, idxs_1):
            data.append(self.data[i:(i+1), j:(j+self.prediction_range+1)])
        return torch.from_numpy(np.concatenate(data, axis=0).astype(np.float32))

    def __len__(self):
        return int(self.data.shape[0]*self.items_per_sample)
