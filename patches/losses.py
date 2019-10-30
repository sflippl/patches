"""Loss functions for patching.
"""

import torch

__all__ = ['Temperature', 'LinearTemperature',
           'ContrastiveLoss', 'BilinearSimilarity']

class Temperature:
    def __init__(self, temperature_fun=None):
        self.temperature_fun = temperature_fun or (lambda x: 1)
        self.it = 0

    def __call__(self):
        temperature = self.temperature_fun(self.it)
        self.it += 1
        return temperature

    def restart(self):
        self.it = 0

class LinearTemperature(Temperature):
    def __init__(self, length):
        if length < 0:
            raise ValueError('Length must be positive.')
        def fun(x):
            if x < length:
                return x/length
            return 1
        super().__init__(fun)

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

    def __init__(self, similarity=None, loss=None, temperature=Temperature()):
        super().__init__()
        self.similarity = similarity or _NegExp(loss)
        self._temperature = temperature 

    def __call__(self, x, temperature=None):
        if temperature is None:
            temperature = self.temperature()
        predicted_code = x['predicted_code'].reshape(x['future_code'].shape)
        true_category = self.similarity(predicted_code,
                                        x['future_code'])
        contrastive_categories = self.similarity(
            predicted_code.repeat_interleave(x['contrastive_code'].shape[-3], axis=-3),
            x['contrastive_code']
        )
        softmax = true_category/(
            true_category + contrastive_categories.sum(axis=-2)
        )
        log_softmax = temperature*torch.log(true_category) - torch.log(
            true_category + contrastive_categories.sum(axis=-2)
        )
        return -log_softmax.mean()

    def temperature(self):
        return self._temperature()

    def restart(self):
        self._temperature.restart()

class BilinearSimilarity(torch.nn.modules.loss._Loss):
    """This is the bilinear similarity function.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, prediction, real_value):
        return torch.exp((prediction*real_value).sum(axis=-1))
