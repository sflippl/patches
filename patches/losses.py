"""Loss functions for patching.
"""

import torch

__all__ = ['ContrastiveLoss', 'BilinearSimilarity']

class _NegExp(torch.nn.modules.loss._Loss):
    """Provides exp(-loss); suitable wrapper around a loss function for the 
    contrastive loss. Do not use on its own!
    """

    def __init__(self, loss):
        super().__init__()
        loss.reduction = 'none'
        self.loss = loss

    def __call__(self, prediction, real_value):
        return torch.exp(-self.loss(prediction, real_value).mean(axis=-1))

class ContrastiveLoss(torch.nn.modules.loss._Loss):
    """Noise contrastive loss. It is assumed that the loss parameter returns a
    sample x timesteps matrix!"""

    def __init__(self, loss=None):
        super().__init__()
        self.loss = loss
        self.loss.reduction = 'none'

    def __call__(self, x):
        predicted_code = x['predicted_code'].reshape(x['future_code'].shape)
        all_codes = torch.cat((x['future_code'], x['contrastive_code']), dim=-3)
        if predicted_code.shape[-3] != all_codes.shape[-3]:
            predicted_code = predicted_code.repeat_interleave(
                all_codes.shape[-3], axis=-3
            )
        criterion = self.loss(predicted_code, all_codes).sum(axis=-1)
        return -torch.nn.LogSoftmax(dim=-2)(-criterion)[:,0,:].mean()

class BilinearLoss(torch.nn.modules.loss._Loss):
    """This is the bilinear similarity function.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, real_value):
        return -(prediction*real_value)
