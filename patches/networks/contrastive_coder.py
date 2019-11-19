"""This module specifies a general contrastive coder.
"""

import torch
import torch.nn as nn

class ContrastiveCoder(nn.Module):
    """A contrastive coder consists of an encoder transforming the input into a latent
    code, and a predictor using that latent code to predict the latent codes of the other
    channels.
    """

    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x):
        """The forward function takes in the contrastive dataset and returns the latent code.
        """
        code = self.encoder(x)
        predicted_code = code[:, 1:]
        prediction = self.predictor(code[:, 0]).reshape(predicted_code.shape)
        return {'target': predicted_code, 'input': prediction}

class ContrastiveAutoCoder(nn.Module):
    """A contrastive autocoder consists of an encoder transforming the input into a latent
    state, an autoregressive process accumulating that latent state and the previous latent code 
    into the current latent code, and a predictor using that latent code to predict future processes.
    """

    def __init__(self, encoder, accumulator, predictor):
        super().__init__()
        self.encoder = encoder
        self.accumulator = accumulator
        self.predictor = predictor

    def forward(self, x, previous_code):
        state = self.encoder(x)
        code = self.accumulator(torch.cat((previous_code, state[:, 0]), dim=-1))
        predicted_state = state[:, 1:]
        prediction = self.predictor(code).reshape(predicted_state.shape)
        return {'target': predicted_state, 'input': prediction}, code
