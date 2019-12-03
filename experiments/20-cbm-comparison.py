import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import patches
import lettertask as lt
from tqdm import tqdm
import os
import argparse

# PREPARE GPU/CPU DEVICE ==========================================

parser = argparse.ArgumentParser(
    description='Experiments on Contrastive Coding'
)
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# MODEL ===========================================================

widths = [
    [5, 5],
    [10, 10],
    [50, 50],
    [5, 10],
    [10, 50],
    [100, 100],
    [5, 5, 5],
    [5, 10, 50],
    [50, 50, 50],
    [100, 100, 100],
    [5, 5, 5, 5, 5],
    [50, 50, 50, 50, 50]
]

change_probabilities = [
    [0.05, 0.5],
    [0.05, 0.2],
    [0.05, 0.1],
    [0.2, 0.5],
    [0.05, 0.1, 0.5],
    [0.05, 0.1, 0.2, 0.4, 0.5]
]

model_combinations = [[width, change_probability]\
                      for width in widths\
                      for change_probability in change_probabilities\
                      if len(width) == len(change_probability)]

samples = [
    100,
    1e4,
    1e5
]

learning_rates = [
    1e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2
]

methods = {'SGD': optim.SGD,
           'Adam': optim.Adam,
           'Adadelta': optim.Adadelta,
           'Adamax': optim.Adamax}

latent_features = range(1, 6)

iterations = 100

epochs = 20

def model_generator(input_features, latent_features, timesteps, **kwargs):
    encoder = nn.Linear(input_features, latent_features)
    predictor = nn.Linear(latent_features, timesteps*latent_features)
    decoder = nn.Linear(timesteps*latent_features, timesteps*input_features)
    return {
        'encoder': encoder,
        'predictor': predictor,
        'decoder': decoder
    }
patchclamp = patches.patchclamp.PatchClamp(model_generator).to(device=args.device)

with tqdm(
    total=len(model_combinations)*len(samples)*len(learning_rates)*len(methods)*len(latent_features)*iterations*epochs
) as pbar:
    for i_model, model in enumerate(model_combinations):
        _latent_features = [lf for lf in latent_features if lf<=len(model[0])]
        for _samples in samples:
            data = lt.data.CompositionalBinaryModel(width=model[0], change_probability=model[1],
                                                    samples=_samples, seed=101)
            ideals = np.diag(1/np.sqrt(np.array(model[0]))).repeat(np.array(model[0]), axis=1)
            for latent_feature in _latent_features:
                for lr in learning_rates:
                    for method_name, method in methods.items():
                        for algorithm in ['scr', 'spcr', 'upa', 'cc']:
                            suffix = '{}_{}_{}_{}_{}_{}'.format(i_model,
                                                                _samples,
                                                                latent_feature,
                                                                lr,
                                                                method_name,
                                                                algorithm)
                            angle_path = '20-data/angles_{}.npy'.format(suffix)
                            loss_path = '20-data/loss_{}.npy'.format(suffix)
                            angle_exists = angle_path in os.listdir('20-data')
                            loss_exists = loss_path in os.listdir('20-data')
                            if angle_exists and loss_exists:
                                continue
                            loss_trajs = []
                            angles = []
                            for it in range(iterations):
                                clamp = patchclamp.get(algorithm,
                                                       input_features = sum(model[0]),
                                                       latent_features = latent_feature,
                                                       timesteps=5)
                                latent_array = data.latent_array()[:, range(latent_feature)]
                                latent_array = latent_array.reshape(1, *latent_array.shape)
                                dataset = clamp.dataset(data.to_array().reshape(1, *data.to_array().shape),
                                                        latent_array,
                                                        timesteps=5)
                                loss_traj = []
                                angle = []
                                optimizer = method(clamp.parameters(), lr=lr)
                                for epoch in range(epochs):
                                    running_loss = 0
                                    if method != 'cc':
                                        dataset_it = iter(dataset)
                                    for i in range(len(dataset)):
                                        if method=='cc':
                                            sample = dataset[i]
                                        else:
                                            sample = next(dataset_it)
                                        optimizer.zero_grad()
                                        model_output = clamp(sample)
                                        loss = clamp.loss(model_output, sample,
                                                          input_loss = nn.MSELoss(),
                                                          latent_loss = patches.losses.BilinearLoss())
                                        loss.backward()
                                        optimizer.step()
                                        if i % 100 == 99:
                                            loss_traj.append(running_loss/100)
                                            params = list(clamp.parameters())[0].detach().numpy()
                                            params = params/np.sqrt((params**2).sum(axis=1))\
                                                              .reshape(params.shape[0], 1)
                                            _angle = np.matmul(
                                                ideals,
                                                params.T
                                            )
                                            angle.append(_angle)
                                    pbar.update(1)
                                angle = np.array(angle)
                                loss_traj = np.array(loss_traj)
                                angles.append(angle)
                                loss_trajs.append(loss_traj)
                            angles = np.array(angles)
                            loss_trajs = np.array(loss_trajs)
                            np.save(angle_path, angles)
                            np.save(loss_path, loss_trajs)
