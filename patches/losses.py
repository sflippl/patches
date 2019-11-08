"""Loss functions for patching.
"""

import torch.nn as nn

__all__ = ['ContrastiveLoss', 'BilinearSimilarity']



class ContrastiveLoss(nn.modules.loss._Loss):
    """The contrastive loss assumes that the first dimension of the passed input and target
    provide the contrasts, and the second one provides the channels.
    """

    def __init__(self, criterion):
        super().__init__()
        criterion.reduction = 'none'
        self.criterion = criterion

    def forward(self, input, target):
        expanded_input = input.reshape(input.shape[0], 1, *input.shape[1:])
        expanded_target = target.expand(1, *target.shape)
        loss = self.criterion(expanded_input, expanded_target).mean(axis=-1)
        loss = -nn.LogSoftmax(dim=1)(-loss)\
                  .mean(axis=tuple(range(2, loss.ndim)))\
                  .diag()\
                  .mean()
        return loss

class BilinearLoss(nn.modules.loss._Loss):
    """This is the bilinear similarity function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return -(input*target)
