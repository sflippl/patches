import lettertask
import patches
import predicode
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import numpy as np
import plotnine as gg
import pandas as pd
from tqdm import tqdm

cifar = predicode.datasets.Cifar10()
cifar_data = cifar.data.transpose((0, 2, 3, 1))\
                       .reshape(50000*32, 3, 32)\
                       .astype(np.float32)/256

cpd = patches.data.Contrastive1DParallelData(
    cifar_data[:,0,:],
    cifar_data[:,1:,:]
)

lst_cpc = []
lf_nrs = range(1, 6)

cpc = patches.networks.LinearScaffold(1, 32, 2)

criterion = patches.losses.ContrastiveLoss(loss=nn.MSELoss())

cdl = utils.data.DataLoader(cpd, batch_size=8)

n_epochs = 10
with tqdm(total = n_epochs*len(cdl)*len(lf_nrs)) as pbar:
    for lf_nr in lf_nrs:
        loss_traj = []
        cpc = patches.networks.LinearScaffold(lf_nr, 32, 2)
        optimizer = optim.Adam(cpc.parameters(), lr=0.01)
        for epoch in range(n_epochs):
            running_loss = 0
            for i, data in enumerate(cdl):
                if i<len(cdl):
                    optimizer.zero_grad()
                    code = cpc(data)
                    loss = criterion(code)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss
                    if i % 5000 == 4999:
                        loss_traj.append(running_loss.detach().numpy()/500)
                        running_loss = 0
                    pbar.update(1)
        np.save('11-data/{}_loss_traj.npy'.format(lf_nr),
                np.array(loss_traj))
        torch.save(cpc, '11-data/{}_cpc.pt'.format(lf_nr))
        