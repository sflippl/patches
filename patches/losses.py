"""Loss functions for patching.
"""

import torch

__all__ = ['ContrastiveLoss']

class _NegExp(torch.nn.modules.loss._Loss):
    """Provides exp(-loss); suitable wrapper around a loss function for the 
    contrastive loss. Do not use on its own!
    """

    def __init__(self, loss):
        super().__init__()
        loss.reduction = 'none'
        self.loss = loss

    def __call__(self, prediction, real_value):
        return torch.exp(-self.loss(prediction, real_value).mean(axis=2))

class ContrastiveLoss(torch.nn.modules.loss._Loss):
    """Noise contrastive loss. It is assumed that the loss parameter returns a
    sample x timesteps matrix!"""

    def __init__(self, similarity=None, loss=None):
        super().__init__()
        self.similarity = similarity or _NegExp(loss)

    def __call__(self, x):
        true_category = self.similarity(x['predicted_code'], x['future_code'])
        contrastive_categories = self.similarity(
            x['predicted_code'].repeat(x['contrastive_code'].shape[0], 1, 1),
            x['contrastive_code']
        )
        softmax = true_category/(
            true_category + contrastive_categories.sum(axis=0)
        )
        return -torch.log(softmax).mean(axis=1)
