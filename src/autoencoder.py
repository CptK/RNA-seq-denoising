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
        encoder_layers = []
        prev_size = input_size
        for h_size in encoder_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = h_size
            
        # Add bottleneck
        encoder_layers.extend([
            nn.Linear(prev_size, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
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
        mean = self.mean_layer(decoded) * size_factors.unsqueeze(1)
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
        mean = self.mean_layer(decoded) * size_factors.unsqueeze(1)
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
        mean = self.mean_layer(decoded) * size_factors.unsqueeze(1)
        dispersion = self.dispersion_layer(decoded)
        dropout = self.dropout_layer(decoded)
        return mean, dispersion, dropout

# Usage example:
model = ZINBAutoencoder(
    input_size=1000,
    encoder_sizes=[512, 256],  # Two encoder layers
    bottleneck_size=128,       # Bottleneck dimension
    decoder_sizes=[256, 512]   # Two decoder layers (symmetric)
)