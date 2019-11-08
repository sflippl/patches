"""Defines the PatchClamp class.
"""

import torch
import torch.nn as nn

class ClampedModel(nn.Module):
    """A method-agnostic syntax for models.

    For a syntax that is agnostic towards the model choice on the spectrum between
    labelled and unlabelled data, this class provides a simple interface for
    computation of the loss and thus subsequent optimization. It works together with
    data from the dataset class 'patches.data.HiddenMarkovModel' and
    'patches.data.TimeSeries'.

    Args:
        module: The network module that is being used for the latent data.
        loss: The loss function used.
        forward_pass: A function that accepts data as an input and outputs the module call.
        loss_pass: A function that accepts the module output and the data as an input,
            and outputs the arguments for the loss.
    """

    def __init__(self, module, model_type=None, forward_pass=None, loss_pass=None):
        self._module = module
        self.forward_pass = forward_pass or ClampedModel._get_forward_pass(model_type)
        self.loss_pass = loss_pass or ClampedModel._get_loss_pass(loss_pass)

    def forward(self, x):
        data = self.forward_pass(x)
        return self._module(data)

    def loss(self, model_output, data, loss):
        loss_arguments = self.loss_pass(model_output, data)
        return loss(**loss_arguments)

    def _get_forward_pass(model_type):
        if (model_type is None) or model_type in ['cc']:
            return lambda data: data
        if model_type in ['scr', 'spcr', 'ura', 'upa']:
            return lambda data: data['input']
        raise ValueError('Unknown model type {}.'.format(model_type))

    def _get_loss_pass(model_type):
        if (model_type is None):
            return lambda model_output, data: {'input': model_output, 'target': data}
        if model_type in ['cc']:
            return lambda model_output, data: model_output
        if model_type in ['scr']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['latents']}
        if model_type in ['spcr']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['predicted_latents']}
        if model_type in ['ura']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['input']}
        if model_type in ['upa']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['predicted_input']}
        raise ValueError('Unknown model type {}.'.format(model_type))

class PatchClamp:
    def __init__(self, model_generator=None, model_hyperparameters=None):
        self.model_generator = model_generator
        self.model_hyperparameters = model_hyperparameters

    def get_scr(self, input_features, latent_features, **kwargs):
        """Get the supervised classification or regression. 
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        scr = ClampedModel(models['encoder'], 'scr')
        return scr

    def get_spcr(self, input_features, latent_features, **kwargs):
        """Get the supervised predictive classification or regression.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        spcr = ClampedModel(nn.Sequential(models['encoder'], models['predictor']), 'spcr')
        return spcr

    def get_cc(self, input_features, latent_features, **kwargs):
        """Get the contrastive coding algorithm.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        cc = ClampedModel(patches.networks.ContrastiveCoding)
