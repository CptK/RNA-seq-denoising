from networks.autoencoder import Autoencoder

import torch.nn as nn

from networks.layers import MeanAct

from networks.loss import poisson_loss


class PoissonAutoencoder(Autoencoder):
    """
    A Poisson Autoencoder neural network.

    This class extends the base Autoencoder class to implement a Poisson Autoencoder,
    which is specifically designed for count data following a Poisson distribution.
    It uses a mean activation function in the output layer and Poisson loss function.

    Attributes:
        Inherits all attributes from the Autoencoder class.

    Note:
        This class overrides the `build_output` and `init_model` methods of the base Autoencoder class.
    """
    def build_output(self):
        """
        Build the output layer of the Poisson Autoencoder.

        This method sets up the mean layer with a Linear layer followed by a MeanAct activation,
        and sets the loss function to Poisson loss.

        Note:
            This method overrides the `build_output` method of the base Autoencoder class.
        """
        self.mean_layer = nn.Sequential(
            nn.Linear(self.decoder_hidden_sizes[-1], self.output_size),
            MeanAct()
        )
        self.loss = poisson_loss

    def init_model(self):
        """
        Initialize the Poisson Autoencoder model.

        This method calls the parent class's `init_model` method and then sets up additional
        model components specific to the Poisson Autoencoder.

        It defines two extra models:
        1. 'mean_norm': Computes the mean of the encoded input.
        2. 'decoded': Returns the encoded input.

        Note:
            This method overrides the `init_model` method of the base Autoencoder class.
        """
        super().init_model()
        self.extra_models['mean_norm'] = lambda x: self.mean_layer(self.encoder(x))
        self.extra_models['decoded'] = lambda x: self.encoder(x)
        