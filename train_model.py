from networks.autoencoder import Autoencoder
from networks.poisson import PoissonAutoencoder
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
from sklearn.model_selection import train_test_split
import pandas as pd

import data

if __name__ == '__main__':
    adata = sc.datasets.paul15()
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    adata = data.read_dataset(adata, transpose=True, test_split=0.1, copy=False, check_counts=False)
    adata = data.normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True)

    train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
    spl = pd.Series(['train'] * adata.n_obs)
    spl.iloc[test_idx] = 'test'
    adata.obs['dca_split'] = spl.values
    output_size = adata.n_vars

    model = PoissonAutoencoder(input_size=output_size)

    # setup data
    validation_split = 0.1
    use_raw_as_output = True
    batch_size = 128
    device =  torch.device("mps")
    X = torch.tensor(adata.X, dtype=torch.float32)
    size_factors = torch.tensor(adata.obs.size_factors.values, dtype=torch.float32).unsqueeze(1)
    output = torch.tensor(adata.raw.X if use_raw_as_output else adata.X, dtype=torch.float32)
    dataset = TensorDataset(X, size_factors, output)
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    loss_fn = model.loss
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    model.to(device)
    for epoch in range(20):
        model.train()
        train_loss = 0
        for batch_x, batch_sf, batch_y in train_loader:
            batch_x, batch_sf, batch_y = batch_x.to(device), batch_sf.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x, batch_sf)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_sf, batch_y in val_loader:
                batch_x, batch_sf, batch_y = batch_x.to(device), batch_sf.to(device), batch_y.to(device)
                output = model(batch_x, batch_sf)
                val_loss += loss_fn(output, batch_y).item()

        val_loss /= len(val_loader)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
