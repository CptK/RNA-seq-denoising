from abc import ABCMeta
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import ColwiseMultLayer


INIT_FUNCTIONS: dict[str, Callable] = {
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "orthogonal": nn.init.orthogonal_,
    "sparse": nn.init.sparse_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
}

def get_init_function(init_name: str) -> Callable:
    """
    Get the initialization function corresponding to the given name.

    Args:
        init_name (str): Name of the initialization method.

    Returns:
        Callable: The corresponding initialization function.

    Raises:
        ValueError: If an unknown initialization method is specified.
    """
    init_function = INIT_FUNCTIONS.get(init_name.lower())
    if init_function is None:
        raise ValueError(f"Unknown initialization method: {init_name}")
    return init_function


class Autoencoder(nn.Module, metaclass=ABCMeta):
    """
    A flexible autoencoder neural network architecture.

    This class implements an autoencoder with customizable encoder and decoder structures,
    including options for dropout, batch normalization, and various activation functions.

    Attributes:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        encoder_hidden_sizes (list[int]): List of hidden layer sizes for the encoder.
        bottleneck_size (int): Size of the bottleneck layer.
        decoder_hidden_sizes (list[int]): List of hidden layer sizes for the decoder.
        input_dropout (float): Dropout rate for the input layer.
        encoder_dropout (float or list[float]): Dropout rates for encoder hidden layers.
        decoder_dropout (float or list[float]): Dropout rates for decoder hidden layers.
        batchnorm (bool): Whether to use batch normalization.
        activation (str): Activation function to use.
        weight_init (str): Weight initialization method.
        file_path (str or None): Path to save/load model weights.
        debug (bool): Whether to enable debug mode.
        loss (Callable): Loss function.
        extra_models (dict): Additional model components.
        model (nn.Module): The complete autoencoder model.
        encoder (nn.Module): The encoder part of the model.
        decoder (nn.Module): The decoder part of the model.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        encoder_hidden_sizes: list[int] = [64],
        bottleneck_size: int = 32,
        decoder_hidden_sizes: list[int] = [64],
        input_dropout: float = 0.0,
        encoder_dropout: float | list[float] = 0.0,
        decoder_dropout: float | list[float] = 0.0,
        batchnorm: bool = True,
        activation: str = "ReLU",
        weight_init: str = "xavier_uniform",
        file_path: str | None = None,
        debug: bool = False
    ):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.bottleneck_size = bottleneck_size
        self.decoder_hidden_sizes = decoder_hidden_sizes
        self.input_dropout = input_dropout
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.weight_init = weight_init
        self.file_path = file_path
        self.debug = debug

        self.loss: Callable = None
        self.extra_models: dict[str, nn.Module] = {}
        self.model: nn.Module = None
        self.encoder: nn.Module = None
        self.decoder: nn.Module = None

        if isinstance(self.encoder_dropout, list):
            assert len(self.encoder_hidden_sizes) == len(self.encoder_dropout)
        else:
            self.encoder_dropout = [self.encoder_dropout] * len(self.encoder_hidden_sizes)

        if isinstance(self.decoder_dropout, list):
            assert len(self.decoder_hidden_sizes) == len(self.decoder_dropout)
        else:
            self.decoder_dropout = [self.decoder_dropout] * len(self.decoder_hidden_sizes)

        if not self.output_size:
            self.output_size = self.input_size
           
        self.init_model()

    def init_model(self):
        """
        Initialize the autoencoder model architecture.

        This method sets up the encoder, bottleneck, and decoder layers of the autoencoder. It also
        initializes additional components like dropout and batch normalization layers.
        """
        self.sf_layer = nn.Identity()

        # Encoder
        layers = []
        next_input_size = self.input_size

        if self.input_dropout > 0:
            layers.append(nn.Dropout(p=self.input_dropout))

        for size, dropout in zip(self.encoder_hidden_sizes, self.encoder_dropout):
            layers.append(nn.Linear(next_input_size, size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(self.get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            next_input_size = size

        self.encoder = nn.Sequential(*layers)

        # Bottleneck
        self.bottleneck = nn.Linear(next_input_size, self.bottleneck_size)

        # Decoder
        layers = []
        next_input_size = self.bottleneck_size
        for size, dropout in zip(self.decoder_hidden_sizes, self.decoder_dropout):
            layers.append(nn.Linear(next_input_size, size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(self.get_activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            next_input_size = size

        self.decoder = nn.Sequential(*layers)
        self.colwise = ColwiseMultLayer()

        self.build_output()

        if self.weight_init == "xavier_uniform":
            self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initialize the weights of the neural network layers.

        This method is used with the apply() function to initialize the weights
        of linear layers using the specified initialization method.

        Args:
            m (nn.Module): A module in the neural network.
        """
        if isinstance(m, nn.Linear):
            self.init_function(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def build_output(self):
        """
        Build the output layer and set up additional model components.

        This method initializes the loss function, creates the final output layer, and sets up extra model
        components for different outputs.
        """
        self.loss = nn.MSELoss()
        self.mean_layer = nn.Linear(self.decoder_hidden_sizes[-1], self.output_size)

        # keep unscaled output as an extra model
        self.extra_models["mean_norm"] = nn.Sequential(self.encoder, self.bottleneck, self.decoder, self.mean_layer)
        self.extra_models["decoded"] = nn.Sequential(self.encoder, self.bottleneck, self.decoder)

    def forward(self, count, size_factors):
        """
        Perform a forward pass through the autoencoder.

        Args:
            count (torch.Tensor): Input data tensor.
            size_factors (torch.Tensor): Size factors for scaling the output.

        Returns:
            torch.Tensor: The output of the autoencoder after applying size factors.
        """
        x = self.encoder(count)
        bottleneck = self.bottleneck(x)
        decoded = self.decoder(bottleneck)
        mean_output = self.mean_layer(decoded)
        output = self.colwise([mean_output, size_factors])
        return output

    def get_activation(self):
        """
        Get the specified activation function.

        Returns:
            nn.Module: The activation function layer based on the specified activation type.

        Raises:
            ValueError: If an unknown activation function is specified.
        """
        if self.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.activation.lower() == 'selu':
            return nn.SELU()
        elif self.activation.lower() == 'elu':
            return nn.ELU()
        elif self.activation.lower() == 'prelu':
            return nn.PReLU()
        elif self.activation.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f'Unknown activation function: {self.activation}')
