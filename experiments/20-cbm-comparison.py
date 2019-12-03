import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import patches
from tqdm import tqdm
import itertools
import os
import argparse

# PREPARE GPU/CPU DEVICE ==========================================

parser = argparse.ArgumentParser(
    description='Experiments on Contrastive Coding'
)
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--iterations',
                    help='Number of runs',
                    type=int,
                    default=100)
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# MODEL ===========================================================

widths = [
    [2, 2],
    [3, 3],
    [7, 7],
    [2, 3],
    [3, 7],
    [10, 10],
    [2, 2, 2],
    [2, 3, 7],
    [7, 7, 7],
    [10, 10, 10],
    [2, 2, 2, 2, 2],
    [7, 7, 7, 7, 7]
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
    encoder = nn.Linear(input_features, latent_features, bias=False)
    predictor = nn.Linear(latent_features, timesteps*latent_features, bias=False)
    decoder = nn.Linear(timesteps*latent_features, timesteps*input_features, bias=False)
    return {
        'encoder': encoder,
        'predictor': predictor,
        'decoder': decoder
    }
patchclamp = patches.patchclamp.PatchClamp(model_generator).to(device=args.device)

i_runs = 0

length = min(args.iterations,
             len(model_combinations)*len(samples)*len(learning_rates)*len(latent_features)*len(methods)*4)

def expandgrid(**itrs):
   product = list(itertools.product(*itrs.values()))
   return pd.DataFrame({key:[x[i] for x in product] for i, key in enumerate(itrs.keys())})

grid = expandgrid(
    _models = zip(range(len(model_combinations)), model_combinations),
    samples = samples,
    latent_features = latent_features,
    _methods = methods.items(),
    algorithm = ['scr', 'spcr', 'upa', 'cc'],
    learning_rates = learning_rates
)
grid['i_model'] = np.array([_model[0] for _model in grid['_models']])
grid['widths'] = [_model[1][0] for _model in grid['_models']]
grid['change_probs'] = [_model[1][1] for _model in grid['_models']]
grid['max_latent_features'] = np.array([len(widths) for widths in grid['widths']])
grid['method_name'] = [_methods[0] for _methods in grid['_methods']]
grid['method'] = [_methods[1] for _methods in grid['_methods']]
grid['latent_features'] = np.array(grid['latent_features'])
grid['samples'] = np.array(grid['samples'], dtype=int)

grid = grid[grid['latent_features'] <= grid['max_latent_features']]

grid['suffix'] = np.array(['{}_{}_{}_{}_{}_{}'.format(
    row.i_model,
    row.samples,
    row.latent_features,
    row.learning_rates,
    row.method_name,
    row.algorithm
) for i, row in grid.iterrows()])

grid['angle_path'] = np.array([
    'angles_{}.npy'.format(suffix) for suffix in grid['suffix']
])

grid['loss_path'] = np.array([
    'loss_{}.npy'.format(suffix) for suffix in grid['suffix']
])

existing_data = os.listdir('20-data')

grid['resolved'] = np.array([
    (row.angle_path in existing_data) and (row.loss_path in existing_data) for i, row in grid.iterrows()
])

print('{}/{} cases resolved. Now resolving another {}.'.format(sum(grid['resolved']),
                                                               len(grid),
                                                               args.iterations))

new_grid = grid[np.logical_not(grid['resolved'])]
new_grid = new_grid.iloc[range(min(len(new_grid), args.iterations))]

with tqdm(total=len(new_grid)*iterations*epochs) as pbar:
    for i, row in new_grid.iterrows():
        data = patches.datasets.compositional_binary_model(widths=row.widths, change_probs=row.change_probs,
                                                    samples=row.samples, seed=101)
        ideals = np.diag(1/np.array(row.widths)).repeat(np.array(row.widths)**2, axis=1)
        loss_trajs = []
        angles = []
        for it in range(iterations):
            clamp = patchclamp.get(row.algorithm,
                                   input_features = sum([_width**2 for _width in row.widths]),
                                   latent_features = row.latent_features,
                                   timesteps=5)
            latent_array = data.latent_array()[:, range(row.latent_features)]
            latent_array = latent_array.reshape(1, *latent_array.shape)
            dataset = clamp.dataset(data.flat_array().reshape(1, row.samples, -1),
                                    latent_array,
                                    timesteps=5)
            loss_traj = []
            angle = []
            optimizer = row.method(clamp.parameters(), lr=row.learning_rates)
            for epoch in range(epochs):
                running_loss = 0
                if row.algorithm != 'cc':
                    dataset_it = iter(dataset)
                for i in range(len(dataset)):
                    if row.algorithm=='cc':
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
                        _angle = np.matmul(ideals, params.T)
                        angle.append(_angle)
                pbar.update(1)
            angle = np.array(angle)
            loss_traj = np.array(loss_traj)
            angles.append(angle)
            loss_trajs.append(loss_traj)
        angles = np.array(angles)
        loss_trajs = np.array(loss_trajs)
        np.save('20-data/{}'.format(row.angle_path,), angles)
        np.save('20-data/{}'.format(row.loss_path,), loss_trajs)

grid['resolved'] = np.array([
    (row.angle_path in existing_data) and (row.loss_path in existing_data) for i, row in grid.iterrows()
])
saved_grid = grid.drop(['_models', '_methods', 'method'], axis='columns')
saved_grid['widths'] = [str(widths) for widths in saved_grid['widths']]
saved_grid['change_probs'] = [str(change_probs) for change_probs in saved_grid['change_probs']]
saved_grid.reset_index(drop=True).to_feather('20-data/grid.feather')
