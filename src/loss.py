import torch
import numpy as np


def _nan2zero(x):
    """
    Replace NaN values in a tensor with zeros.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with NaN values replaced by zeros.
    """
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    """
    Replace NaN values in a tensor with positive infinity.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with NaN values replaced by positive infinity.
    """
    return torch.where(torch.isnan(x), torch.full_like(x, np.inf), x)


def _nelem(x):
    """
    Count the number of non-NaN elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Number of non-NaN elements. Returns 1 if all elements are NaN.
    """
    nelem = torch.sum(~torch.isnan(x)).float()
    return torch.where(nelem == 0., torch.tensor(1., device=x.device), nelem).to(x.dtype)


def _reduce_mean(x):
    """
    Compute the mean of non-NaN elements in a tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Mean of non-NaN elements.
    """
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.sum(x) / nelem

class Poisson:
    def loss(self, y_true, y_pred):
        """
        Compute the Poisson loss between true and predicted values.

        This loss is suitable for count data following a Poisson distribution. It handles NaN values in the true
        values.

        Args:
            y_true (torch.Tensor): True values.
            y_pred (torch.Tensor): Predicted values.

        Returns:
            torch.Tensor: Scalar tensor containing the mean Poisson loss.

        Note:
            The function adds a small epsilon (1e-10) to the predicted values to prevent log(0) errors.
        """
        y_pred = y_pred.float()
        y_true = y_true.float()

        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)

        ret = y_pred - y_true * torch.log(y_pred + 1e-10) + torch.lgamma(y_true + 1.0)

        return torch.sum(ret) / nelem
    
    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


class NB:
    """
    Negative Binomial (NB) distribution loss function.

    This class implements the negative log-likelihood loss of the Negative Binomial distribution, which is
    useful for modeling overdispersed count data in various machine learning tasks.

    Attributes:
        theta (float or torch.Tensor): Dispersion parameter of the NB distribution.
        masking (bool): If True, enables masking of missing values.
        scope (str): Scope name for the loss function.
        scale_factor (float): Scaling factor for predictions.
        debug (bool): If True, enables additional debugging assertions.
        eps (float): Small value to prevent division by zero and log(0) errors.

    """
    def __init__(self, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking

    def loss(self, y_true, y_pred, theta, mean=True):
        """
        Compute the Negative Binomial loss between true and predicted values.

        This method calculates the negative log-likelihood of the Negative Binomial distribution.

        Args:
            y_true (torch.Tensor): True values.
            y_pred (torch.Tensor): Predicted values.
            mean (bool, optional): If True, return the mean loss across all elements. 
                                   If False, return the loss for each element. Defaults to True.

        Returns:
            torch.Tensor: Computed NB loss.

        Notes:
            - The method includes provisions for numerical stability and handling of edge cases.
            - If masking is enabled, it handles missing values in the input.
        """
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = y_true.float()
        y_pred = y_pred.float() * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        theta = torch.clamp(theta, max=1e6)

        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))

        if self.debug:
            assert torch.isfinite(y_pred).all(), 'y_pred has inf/nans'
            assert torch.isfinite(t1).all(), 't1 has inf/nans'
            assert torch.isfinite(t2).all(), 't2 has inf/nans'

        final = t1 + t2
        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.sum(final) / nelem
            else:
                final = torch.mean(final)

        return final
    
    def __call__(self, y_true, y_pred, theta):
        return self.loss(y_true, y_pred, theta)


class ZINB(NB):
    """
    Zero-Inflated Negative Binomial (ZINB) distribution loss function.

    This class extends the Negative Binomial (NB) class to implement the ZINB model,
    which is useful for modeling overdispersed count data with excess zeros.

    Attributes:
        pi (float or torch.Tensor): Mixture parameter for the zero-inflation component.
        ridge_lambda (float): Ridge regression parameter for regularization.

    Inherits all attributes from the NB class.
    """
    def __init__(self, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, theta, pi, mean=True):
        """
        Compute the Zero-Inflated Negative Binomial loss between true and predicted values.

        This method calculates the negative log-likelihood of the ZINB distribution, which combines the
        Negative Binomial distribution with a point mass at zero.

        Args:
            y_true (torch.Tensor): True values.
            y_pred (torch.Tensor): Predicted values.
            mean (bool, optional): If True, return the mean loss across all elements. 
                                   If False, return the loss for each element. Defaults to True.

        Returns:
            torch.Tensor: Computed ZINB loss.

        Notes:
            - The method handles both the zero and non-zero components of the ZINB model.
            - It includes a ridge regression term for regularization.
            - The method includes provisions for numerical stability and handling of edge cases.
            - If masking is enabled (inherited from NB), it handles missing values in the input.
        """
        scale_factor = self.scale_factor
        eps = self.eps

        nb_case = super().loss(y_true, y_pred, theta, mean=False) - torch.log(1.0 - pi + eps)

        y_true = y_true.float()
        y_pred = y_pred.float() * scale_factor
        theta = torch.clamp(theta, max=1e6)

        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.mean(result)

        result = _nan2inf(result)

        return result
    
    def __call__(self, y_true, y_pred, theta, pi):
        return self.loss(y_true, y_pred, theta, pi)
    

class ZINBVariational(ZINB):
    def __init__(self, beta=1.0, ridge_lambda=0.0, scope='zinb_vae_loss/', **kwargs):
        super().__init__(ridge_lambda=ridge_lambda, scope=scope, **kwargs)
        self.beta = beta  # KL weight factor

    def loss(self, y_true, y_pred, z_mean, z_log_var, theta, pi, mean=True):
        """
        Compute combined ZINB and KL loss.
        
        Args:
            y_true (torch.Tensor): True values
            y_pred (torch.Tensor): Predicted values
            z_mean (torch.Tensor): Mean of latent space
            z_log_var (torch.Tensor): Log variance of latent space
            mean (bool): Whether to return mean loss
        """
        reconstruction_loss = super().loss(y_true, y_pred, theta, pi, mean=mean)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        
        if mean:
            kl_loss = kl_loss / y_true.size(0) # Average over batch
            
        return reconstruction_loss + self.beta * kl_loss
    
    def __call__(self, y_true, y_pred, z_mean, z_log_var, theta, pi):
        return self.loss(y_true, y_pred, z_mean, z_log_var, theta, pi)
