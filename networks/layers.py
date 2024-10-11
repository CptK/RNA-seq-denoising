import torch
import torch.nn as nn

class ColwiseMultLayer(nn.Module):
    """
    A custom PyTorch layer that performs column-wise multiplication of two tensors.

    This layer is typically used to apply size factors or other scaling factors to the output of a neural
    network.

    Args:
        None

    Input:
        x (tuple): A tuple containing two tensors:
            - tensor1 (torch.Tensor): The primary input tensor of shape (batch_size, features).
            - tensor2 (torch.Tensor): The scaling factor tensor of shape (batch_size,).

    Returns:
        torch.Tensor: The result of element-wise multiplication of tensor1 with
                      tensor2 expanded to match tensor1's shape.

    Shape:
        - Input: ((N, D), (N,))
        - Output: (N, D)
        where N is the batch size and D is the number of features.
    """
    def forward(self, x):
        tensor1, tensor2 = x
        return tensor1 * tensor2.unsqueeze(1)
    
class MeanAct(nn.Module):
    """
    A custom PyTorch activation layer that applies an exponential function followed by clamping.

    This activation is often used in models dealing with count data or non-negative outputs, such as in
    certain types of autoencoders for genomic data. It allows for customizable minimum and maximum output 
    values.

    Args:
        min (float, optional): The minimum value for clamping. Default is 1e-5.
        max (float, optional): The maximum value for clamping. Default is 1e6.

    Input:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The result of exp(x) clamped between the specified min and max values.

    Shape:
        - Input: (N, *)
        - Output: (N, *)
        where * means any number of additional dimensions.
    """
    def __init__(self, min=1e-5, max=1e6):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=self.min, max=self.max)