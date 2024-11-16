import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseAutoencoder(nn.Module, ABC):
    def __init__(
        self, 
        input_size: int,
        encoder_sizes: list[int],
        bottleneck_size: int,
        decoder_sizes: list[int],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        
        # Build encoder
        self.encoder_layers = []
        prev_size = input_size
        for h_size in encoder_sizes:
            self.encoder_layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = h_size
            
        # Add bottleneck
        self.encoder_layers.extend([
            nn.Linear(prev_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_size = bottleneck_size
        for h_size in decoder_sizes:
            decoder_layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = h_size
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.decoder_output_size = prev_size
        
        # Output layers will be implemented by subclasses
        self._build_output_layers()

    @abstractmethod
    def _build_output_layers(self):
        """Build the output layers specific to each distribution"""
        pass

    @abstractmethod
    def forward(self, x, size_factors):
        """Forward pass implementation specific to each distribution"""
        pass

    def get_encoded(self, x):
        """Get the bottleneck representation"""
        return self.encoder(x)

class PoissonAutoencoder(BaseAutoencoder):
    def _build_output_layers(self):
        self.mean_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Softplus()
        )

    def forward(self, x, size_factors):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        mean = self.mean_layer(decoded) * size_factors
        return mean

class NBAutoencoder(BaseAutoencoder):
    def _build_output_layers(self):
        self.mean_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Softplus()
        )
        self.dispersion_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Softplus()
        )

    def forward(self, x, size_factors):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        mean = self.mean_layer(decoded) * size_factors
        dispersion = self.dispersion_layer(decoded)
        return mean, dispersion

class ZINBAutoencoder(BaseAutoencoder):
    def _build_output_layers(self):
        self.mean_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Softplus()
        )
        self.dispersion_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Softplus()
        )
        self.dropout_layer = nn.Sequential(
            nn.Linear(self.decoder_output_size, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x, size_factors):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        mean = self.mean_layer(decoded) * size_factors
        dispersion = self.dispersion_layer(decoded)
        dropout = self.dropout_layer(decoded)
        return mean, dispersion, dropout


class ZINBVariationalAutoEnocder(ZINBAutoencoder):
    def __init__(
        self, 
        input_size: int,
        encoder_sizes: list[int],
        bottleneck_size: int,
        decoder_sizes: list[int],
        dropout_rate: float = 0.1
    ):
        super().__init__(input_size, encoder_sizes, bottleneck_size, decoder_sizes, dropout_rate)
        self.encoder_layers = self.encoder_layers[:-4]
        self.encoder = nn.Sequential(*self.encoder_layers)

        last_encoder_size = encoder_sizes[-1]
        self.z_mean = nn.Linear(last_encoder_size, bottleneck_size)
        self.z_log_var = nn.Linear(last_encoder_size, bottleneck_size)


    def forward(self, x, size_factors):
        encoded = self.encoder(x)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        z = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(z)
        mean = self.mean_layer(decoded) * size_factors
        dispersion = self.dispersion_layer(decoded)
        dropout = self.dropout_layer(decoded)
        return mean, dispersion, dropout, z, z_mean, z_log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


# Usage example:
model = ZINBAutoencoder(
    input_size=1000,
    encoder_sizes=[512, 256],  # Two encoder layers
    bottleneck_size=128,       # Bottleneck dimension
    decoder_sizes=[256, 512]   # Two decoder layers (symmetric)
)