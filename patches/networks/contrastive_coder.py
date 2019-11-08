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
