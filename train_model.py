import scanpy as sc
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, random_split
from torch import optim
import numpy as np
from tqdm import tqdm

from src.autoencoder import NBAutoencoder, ZINBAutoencoder, PoissonAutoencoder
from src.loss import poisson_loss, NB, ZINB  # poisson_loss is a function, NB and ZINB are classes
from src.data import read_dataset, normalize

import matplotlib.pyplot as plt


USE_RAW_OUTPUT: bool = False
BATCH_SIZE: int = 32
VAL_SPLIT: float = 0.2
DEVICE = 'mps'
EPOCHS: int = 300
MODEL_TYPE: str = 'zinb'
LR: float = 1e-3
ENCODER_SIZES: list[int] = [64]  # Hidden layer sizes for encoder, [size_layer1, size_layer2, ...]
DECODER_SIZES: list[int] = [64]  # Hidden layer sizes for decoder, [size_layer1, size_layer2, ...]
BOTTLNECK_SIZE: int = 32  # Size of bottleneck layer
DROPOUT_RATE: float = 0.1  # Dropout rate for encoder and decoder


def get_model(model_type: str, input_dim: int, encoder_sizes, bottleneck_size, decoder_sizes, dropout_rate):
    if model_type == 'zinb':
        model = ZINBAutoencoder(
            input_size=input_dim,
            encoder_sizes=encoder_sizes,
            bottleneck_size=bottleneck_size,
            decoder_sizes=decoder_sizes,
            dropout_rate=dropout_rate
        )
    elif model_type == 'nb':
        model = NBAutoencoder(
            input_size=input_dim,
            encoder_sizes=encoder_sizes,
            bottleneck_size=bottleneck_size,
            decoder_sizes=decoder_sizes,
            dropout_rate=dropout_rate
        )
    else:
        model = PoissonAutoencoder(
            input_size=input_dim,
            encoder_sizes=encoder_sizes,
            bottleneck_size=bottleneck_size,
            decoder_sizes=decoder_sizes,
            dropout_rate=dropout_rate
        )
    return model


def train_epoch(model, train_loader, optimizer, epoch, epochs, device):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
        for batch_x, batch_sf, batch_y in pbar:
            # Move to device
            batch_x = batch_x.to(device)
            batch_sf = batch_sf.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            if isinstance(model, ZINBAutoencoder):
                mean, disp, pi = model(batch_x, batch_sf)
                zinb = ZINB(pi=pi, theta=disp)
                loss = zinb.loss(batch_y, mean)
            elif isinstance(model, NBAutoencoder):
                mean, disp = model(batch_x, batch_sf)
                nb = NB(theta=disp)
                loss = nb.loss(batch_y, mean)
            else:  # PoissonAutoencoder
                mean = model(batch_x, batch_sf)
                loss = poisson_loss(batch_y, mean)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def train_model(model, epochs, train_loader, val_loader, optimizer, scheduler, device):
    best_val_loss = float('inf')
    early_stopping_patience = 20
    early_stopping_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, epochs, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'best_model.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

    return best_val_loss, train_losses, val_losses


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    for batch_x, batch_sf, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_sf = batch_sf.to(device)
        batch_y = batch_y.to(device)
        
        if isinstance(model, ZINBAutoencoder):
            mean, disp, pi = model(batch_x, batch_sf)
            zinb = ZINB(pi=pi, theta=disp)
            loss = zinb.loss(batch_y, mean)
        elif isinstance(model, NBAutoencoder):
            mean, disp = model(batch_x, batch_sf)
            nb = NB(theta=disp)
            loss = nb.loss(batch_y, mean)
        else:
            mean = model(batch_x, batch_sf)
            loss = poisson_loss(batch_y, mean)
            
        total_loss += loss.item()
    
    return total_loss / len(val_loader)


if __name__ == '__main__':
    # Load the Paul et al. dataset
    adata = sc.datasets.paul15()
    print(f"Finished loading dataset: {adata.X.shape}")

    # Preprocess the data
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    adata = read_dataset(adata=adata, transpose=True, test_split=True, check_counts=False)
    adata = normalize(adata=adata, size_factors=True, logtrans_input=True, normalize_input=True)

    X = torch.tensor(adata.X, dtype=torch.float32)

    size_factors = torch.tensor(adata.obs["size_factors"].values, dtype=torch.float32).unsqueeze(1)
    output = torch.tensor(adata.raw.X if USE_RAW_OUTPUT else adata.X, dtype=torch.float32)

    dataset = TensorDataset(X, size_factors, output)
    train_size = int((1 - VAL_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X.shape[1]
    model = get_model(MODEL_TYPE, input_dim, ENCODER_SIZES, BOTTLNECK_SIZE, DECODER_SIZES, DROPOUT_RATE)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss, train_losses, val_losses = train_model(
        model, EPOCHS, train_loader, val_loader, optimizer, scheduler, DEVICE
    )
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Losses for {MODEL_TYPE} model")
    plt.legend()
    plt.show()
                
    # Load best model
    checkpoint = torch.load(f'best_model.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create a loader for the full dataset
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get latent representations for full dataset
    model.eval()
    all_latent = []
    with torch.no_grad():
        for batch_x, batch_sf, _ in full_loader:
            batch_x = batch_x.to(DEVICE)
            batch_sf = batch_sf.to(DEVICE)
            latent = model.get_encoded(batch_x)
            all_latent.append(latent.cpu().numpy())

    latent = np.concatenate(all_latent, axis=0)

    # Add to AnnData object
    adata.obsm[f'X_{MODEL_TYPE}_latent'] = latent

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
