"""Implements a simple predictable scaffold.
"""

import abc

import torch
import torch.nn as nn
import numpy as np

__all__ = ['SimpleScaffold', 'LinearScaffold']

class SimpleScaffold(nn.Module, abc.ABC):
    """The simple scaffold determines the latent layer by an encoder and a
    predictor that must be specified in the methods. As such, the simple
    scaffold itself is an abstract base class."""

    @abc.abstractmethod
    def encode(self, x):
        """Here, you specify how values are being encoded.
        """

    @abc.abstractmethod
    def predict(self, x):
        """Here, you specify how prediction on the level of the latents works.
        """

    def forward(self, x):
        """The forward method accepts a ContrastiveTSDataset and returns the
        predicted as well as the future and contrastive codes.
        """
        current_code = self.encode(x['current_values'])
        future_code = self.encode(x['future_values'])
        contrastive_code = self.encode(x['contrastive_values'])
        predicted_code = self.predict(current_code)
        return {
            'future_code': future_code,
            'contrastive_code': contrastive_code,
            'predicted_code': predicted_code
        }

class LinearScaffold(SimpleScaffold):
    """The linear scaffold consists of a linear layer encoding n features and
    a linear predictor predicting those n features for a number of timesteps.
    You can either specify the number of input features and the number of
    timesteps or provide a Contrastive1DTimeSeries to infer those automatically.
    """

    def __init__(self, latent_features, input_features=None, timesteps=None,
                 data=None, bias=True):
        super().__init__()
        if data:
            input_features = input_features or data.n_vars
            timesteps = timesteps or data.n_timesteps
        elif input_features is None or timesteps is None:
            raise ValueError('You must either provide data or both input '
                             'features and timesteps.')
        self.latent_features = latent_features
        self.input_features = input_features
        self.timesteps = timesteps
        self.encoder = nn.Linear(input_features, latent_features, bias=bias)
        self.predictor = nn.Linear(latent_features, latent_features * timesteps,
                                   bias=bias)

    def encode(self, x):
        return self.encoder(x)

    def predict(self, x):
        flat_prediction = self.predictor(x)
        return flat_prediction.reshape(flat_prediction.shape[0],
                                       self.timesteps,
                                       self.latent_features)
