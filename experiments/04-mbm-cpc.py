"""Mapping out different CBMs with different numbers and different paradigms.
"""

import patches
import lettertask
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import plotnine as gg
import lazytools_sflippl as lazytools
import torch.nn as nn

# DATA =========================================================================

cbms = [
    lettertask.data.CompositionalBinaryModel(
        width=width, change_probability=change_probability,
        samples=10000, seed=2002
    ) for width, change_probability in zip(
        [[5, 5], [5, 5], [50, 50], [50, 50],
         [5, 10], [10, 50], [5, 5, 5], [10, 10, 10]],
        [[0.05, 0.5], [0.05, 0.2], [0.05, 0.5], [0.05, 0.2],
         [0.2, 0.05], [0.2, 0.1], [0.05, 0.1, 0.5], [0.05, 0.1, 0.5]]
    )
]

# OPTIMIZATION REGIMES =========================================================

def regime(method, lr):
    def optimizer(params):
        return method(params, lr=lr)
    return optimizer
optimization_regimes = []
for method in [optim.SGD, optim.Adam, optim.Adadelta]:
    for lr in [1e-2, 1e-1, 1]:
        optimization_regimes.append(regime(method, lr))

loss_dfs = []
angle_dfs = []
n_epochs = 5

# NEURAL NETWORKS ==============================================================

class BaRec(nn.Module):
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
        self.predictor = nn.Linear(latent_features, timesteps, bias=bias)
        self.decoder = nn.Conv1d(latent_features, input_features, 1, bias=bias)

    def forward(self, x):
        code = self.encoder(x['current_values'])
        prediction = self.predictor(code)
        decoded = self.decoder(prediction).transpose(1, 2)
        return decoded

class LaPred1P(nn.Module):
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
        self.predictor = nn.Linear(latent_features, timesteps*latent_features,
                                   bias=bias)

    def forward(self, x):
        code = self.encoder(x['input'])
        prediction = self.predictor(code).\
                          reshape(self.timesteps, self.latent_features)
        return prediction

class LaPred2P(nn.Module):
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
        self.predictor = nn.Linear(latent_features, timesteps*latent_features,
                                   bias=bias)

    def forward(self, x):
        code = self.encoder(x['input'])
        prediction = self.predictor(x['latent_values']).\
                          reshape(self.timesteps, self.latent_features)
        return {
            'latent_values': code,
            'latent_prediction': prediction
        }

# MAIN PART ====================================================================

with tqdm(total = n_epochs*len(optimization_regimes)*len(cbms)*4) as pbar:
    for idx_cbm, cbm in enumerate(cbms):
        ideal = np.identity(len(cbm.width)).repeat(cbm.width, 1)
        for idx_opt, opt in enumerate(optimization_regimes):
            cts = patches.data.Contrastive1DTimeSeries(cbm.to_array(), seed=202)

            # BaRec ============================================================

            ## Prepare =========================================================

            barec = BaRec(1, data=cts)
            optimizer = opt(barec.parameters())
            criterion = nn.MSELoss()
            loss_traj = []
            angles = []
            running_loss = 0

            ## Fit =============================================================

            for epoch in range(n_epochs):
                for i, data in enumerate(cts):
                    if i<len(cts):
                        if i % 10 == 0:
                            est = next(barec.parameters()).detach().numpy()
                            angles.append(np.matmul(ideal, est.T)/
                                          np.sqrt(np.matmul(est, est.T)))
                        optimizer.zero_grad()
                        prediction = barec(data)
                        loss = criterion(prediction, data['future_values'])
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                        if i % 50 == 49:
                            loss_traj.append(running_loss.detach().numpy()/50)
                            running_loss = 0
                pbar.update(1)

            ## Cleanup =========================================================

            loss_traj = np.array(loss_traj)
            angles = np.concatenate(angles, axis=1)
            np.save('04-data/arrays/cbm_{}_opt_{}_barec_loss.npy'.\
                    format(idx_cbm, idx_opt), loss_traj)
            np.save('04-data/arrays/cbm_{}_opt_{}_barec_angles.npy'.\
                    format(idx_cbm, idx_opt), angles)
            hmm = patches.data.HiddenMarkovModel(cbm.to_array(),
                                                 cbm.latent_array()[:, [0]])

            # LaPred1P =========================================================

            ## Prepare =========================================================

            lapred1p = LaPred1P(1, data=hmm, bias=False)
            optimizer = opt(lapred1p.parameters())
            criterion = nn.MSELoss()
            running_loss = 0
            loss_traj = []
            angles = []

            ## Fit =============================================================

            for epoch in range(n_epochs):
                for i, data in enumerate(hmm):
                    if i<len(hmm):
                        if i % 10 == 0:
                            est = list(lapred1p.parameters())[0].detach()\
                                                                .numpy()
                            angles.append(np.matmul(ideal, est.T)/
                                          np.sqrt(np.matmul(est, est.T)))
                        optimizer.zero_grad()
                        prediction = lapred1p(data)
                        loss = criterion(prediction,
                                         data['future_latent_values'])
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                        if i % 50 == 49:
                            loss_traj.append(running_loss.detach().numpy()/50)
                            running_loss = 0
                pbar.update(1)

            ## Cleanup =========================================================

            loss_traj = np.array(loss_traj)
            angles = np.concatenate(angles, axis=1)
            np.save('04-data/arrays/cbm_{}_opt_{}_lapred1p_loss.npy'.\
                    format(idx_cbm, idx_opt), loss_traj)
            np.save('04-data/arrays/cbm_{}_opt_{}_lapred1p_angles.npy'.\
                    format(idx_cbm, idx_opt), angles)

            # LaPred2P =========================================================

            ## Prepare =========================================================

            lapred2p = LaPred2P(1, data=hmm, bias=False)
            optimizer = opt(lapred2p.parameters())
            criterion = nn.MSELoss()
            loss_traj = []
            angles = []
            running_loss = 0

            ## Fit =============================================================

            for epoch in range(n_epochs):
                for i, data in enumerate(hmm):
                    if i<len(hmm):
                        if i % 10 == 0:
                            est = list(lapred2p.parameters())[0].detach()\
                                                                .numpy()
                            angles.append(np.matmul(ideal, est.T)/
                                          np.sqrt(np.matmul(est, est.T)))
                        optimizer.zero_grad()
                        prediction = lapred2p(data)
                        loss = criterion(prediction['latent_prediction'],
                                         data['future_latent_values']) + \
                               criterion(prediction['latent_values'],
                                         data['latent_values'])
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                        if i % 50 == 49:
                            loss_traj.append(running_loss.detach().numpy()/50)
                            running_loss = 0
                pbar.update(1)

            ## Cleanup =========================================================

            loss_traj = np.array(loss_traj)
            angles = np.concatenate(angles, axis=1)
            np.save('04-data/arrays/cbm_{}_opt_{}_lapred2p_loss.npy'.\
                    format(idx_cbm, idx_opt), loss_traj)
            np.save('04-data/arrays/cbm_{}_opt_{}_lapred2p_angles.npy'.\
                    format(idx_cbm, idx_opt), angles)

            # Linear Scaffold ==================================================

            ## Prepare =========================================================

            cts = patches.data.Contrastive1DTimeSeries(data=cbm.to_array())
            ce = patches.networks.LinearScaffold(latent_features=1, data=cts)
            criterion = patches.losses.ContrastiveLoss(loss=nn.MSELoss())
            optimizer = opt(ce.parameters())
            angles = []
            loss_traj = []
            running_loss = 0

            ## Fit =============================================================

            for epoch in range(n_epochs):
                for i, data in enumerate(cts):
                    if i<len(cts):
                        if i % 10 == 0:
                            est = list(ce.parameters())[0].detach().numpy()
                            angles.append(np.matmul(ideal, est.T)/
                                          np.sqrt(np.matmul(est, est.T)))
                        optimizer.zero_grad()
                        code = ce(data)
                        loss = criterion(code)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss
                        if i % 50 == 49:
                            loss_traj.append(running_loss.detach().numpy()/50)
                            running_loss = 0
                pbar.update(1)

            ## Cleanup =========================================================

            loss_traj = np.array(loss_traj)
            angles = np.concatenate(angles, axis=1)
            np.save('04-data/arrays/cbm_{}_opt_{}_cpc_loss.npy'.\
                    format(idx_cbm, idx_opt), loss_traj)
            np.save('04-data/arrays/cbm_{}_opt_{}_cpc_angles.npy'.\
                    format(idx_cbm, idx_opt), angles)
