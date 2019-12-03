"""Defines the PatchClamp class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import patches.networks as networks
import patches.losses as losses
import patches.data as data

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

    def __init__(self, module, model_type=None, input_data=None, latent_data=None,
                 forward_pass=None, loss_pass=None,
                 loss_to_criterion=None, timesteps=[0], device=None, **kwargs):
        super().__init__()
        self._module = module
        self._module.to(device=device)
        self.model_type = model_type
        self.forward_pass = forward_pass or ClampedModel._get_forward_pass(model_type)
        self.loss_pass = loss_pass or ClampedModel._get_loss_pass(model_type)
        self.loss_to_criterion = loss_to_criterion or \
                                 ClampedModel._get_loss_to_criterion(model_type)
        self.device = device

    def to(self, device=None):
        self.device = device
        self._module.to(device=device)
        return self

    def forward(self, x):
        data = self.forward_pass(x)
        data = self._module(data)
        if self.model_type == 'upa':
            data = data.reshape(*x['target'].shape)
        return data

    def loss(self, model_output, data, loss=None, input_loss=None, latent_loss=None):
        input_loss = input_loss or loss
        latent_loss = latent_loss or loss
        loss_arguments = self.loss_pass(model_output=model_output, data=data)
        criterion = self.loss_to_criterion(loss, input_loss, latent_loss)
        return criterion(**loss_arguments).mean()

    def _get_forward_pass(model_type):
        if (model_type is None) or model_type in ['cc']:
            return lambda data: data
        if model_type in ['scr', 'spcr']:
            return lambda data: data['input']
        if model_type in ['ura', 'upa']:
            return lambda data: data['input'].reshape(-1, data['input'].shape[-1])
        raise ValueError('Unknown model type {}.'.format(model_type))

    def _get_loss_pass(model_type):
        if (model_type is None):
            return lambda model_output, data: {'input': model_output, 'target': data}
        if model_type in ['cc']:
            return lambda model_output, data: model_output
        if model_type in ['scr']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['target']}
        if model_type in ['spcr']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['target']}
        if model_type in ['ura']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['input']}
        if model_type in ['upa']:
            return lambda model_output, data: {'input': model_output,
                                               'target': data['target']}
        raise ValueError('Unknown model type {}.'.format(model_type))

    def _get_loss_to_criterion(model_type):
        if (model_type is None) or (model_type in ['ura', 'upa', 'scr', 'spcr']):
            return lambda loss, input_loss, latent_loss: input_loss
        if model_type in ['cc']:
            return lambda loss, input_loss, latent_loss: losses.ContrastiveLoss(latent_loss)       
        raise ValueError('Unknown model type {}.'.format(model_type))

    def dataset(self, input_data, latent_data, timesteps, batch_size=8, **kwargs):
        if self.model_type in ['ura']:
            dataset = data.Timeseries(input_data, timesteps=[0])
            return DataLoader(dataset, batch_size=batch_size, drop_last=True)
        if self.model_type in ['upa']:
            dataset = data.Timeseries(input_data, timesteps=range(1, timesteps+1))
            return DataLoader(dataset, batch_size=batch_size, drop_last=True)
        if self.model_type in ['scr']:
            dataset = data.HiddenMarkovModel(input_data, latent_data, timesteps=[0])
            return DataLoader(dataset, batch_size=batch_size, drop_last=True)
        if self.model_type in ['spcr']:
            dataset = data.HiddenMarkovModel(input_data, latent_data, timesteps=range(1, timesteps+1))
            return DataLoader(dataset, batch_size=batch_size, drop_last=True)
        if self.model_type in ['cc']:
            return data.ContrastiveDataset(input_data, contrast_size=9, prediction_range=timesteps,
                                           contrast_type='both')
        raise ValueError('Unknown model type {}.'.format(self.model_type))

class Transpose(nn.Module):
    """Transposing layer for use in sequential.
    """

    def __init__(self, dim0, dim1):
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class PatchClamp:
    class Predictor(nn.Module):
        """Transposing the latter two dimensions in the predictor.
        """

        def __init__(self, encoder, predictor, timesteps=1, latent_features=-1):
            super().__init__()
            self.encoder = encoder
            self.predictor = predictor
            self.timesteps = timesteps
            self.latent_features = latent_features

        def forward(self, x):
            x = self.encoder(x)
            x = self.predictor(x)
            x = x.reshape(*x.shape[:-2], self.timesteps, self.latent_features)
            return x

    def __init__(self, model_generator=None, model_hyperparameters=None, device=None):
        self.model_generator = model_generator
        self.model_hyperparameters = model_hyperparameters
        self.device = device

    def to(self, device=None):
        self.device = device
        return self

    def get_scr(self, input_features, latent_features, **kwargs):
        """Get the supervised classification or regression. 
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        scr = ClampedModel(models['encoder'], 'scr', device=self.device)
        return scr

    def get_spcr(self, input_features, latent_features, timesteps=1, **kwargs):
        """Get the supervised predictive classification or regression.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      timesteps=timesteps,
                                      **kwargs)
        spcr = ClampedModel(PatchClamp.Predictor(models['encoder'],
                                                 models['predictor'],
                                                 timesteps=timesteps,
                                                 latent_features=latent_features),
                            'spcr', device=self.device)
        return spcr

    def get_cc(self, input_features, latent_features, **kwargs):
        """Get the contrastive coding algorithm.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        cc = ClampedModel(networks.ContrastiveCoder(models['encoder'], models['predictor']),
                          'cc', device=self.device)
        return cc

    def get_ura(self, input_features, latent_features, **kwargs):
        """Get the unsupervised reconstructive algorithm.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      **kwargs)
        ura = ClampedModel(nn.Sequential(models['encoder'], models['decoder']), 'ura',
                           device=self.device)
        return ura

    def get_upa(self, input_features, latent_features, timesteps=1, **kwargs):
        """Get the unsupervised predictive algorithm.
        """
        models = self.model_generator(input_features=input_features,
                                      latent_features=latent_features,
                                      timesteps=timesteps,
                                      **kwargs)
        upa = ClampedModel(nn.Sequential(models['encoder'],
                                         models['predictor'],
                                         models['decoder']),
                           'upa', device=self.device)
        return upa

    def get(self, algorithm, **kwargs):
        if algorithm=='scr':
            return self.get_scr(**kwargs)
        if algorithm=='spcr':
            return self.get_spcr(**kwargs)
        if algorithm=='ura':
            return self.get_ura(**kwargs)
        if algorithm=='upa':
            return self.get_upa(**kwargs)
        if algorithm=='cc':
            return self.get_cc(**kwargs)
        raise ValueError('Unknown algorithm {}.'.format(algorithm))
