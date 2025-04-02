import numpy as np 
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, silhouette_score
from scipy.spatial.distance import pdist, squareform


def evaluate_denoising_quality(original, noisy, denoised, labels):
    """
    Calculate multiple metrics to evaluate denoising quality.
    
    Parameters:
    -----------
    original : np.ndarray
        Original (ground truth) data matrix 
    noisy : np.ndarray
        Data matrix with technical noise
    denoised : np.ndarray
        Denoised data matrix
    labels : np.ndarray
        Array of cell type labels for silhouette coefficient calculation
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # 1. Basic Error Metrics
    metrics['mse'] = mean_squared_error(original, denoised)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mse_between_original_and_noisy'] = mean_squared_error(original, noisy)
    
    # 2. Improvement Metrics
    noisy_mse = mean_squared_error(original, noisy)
    metrics['mse_improvement'] = (noisy_mse - metrics['mse']) / noisy_mse * 100
    
    # 3. Correlation Based Metrics 
    # Row-wise correlations (across features)
    row_corrs = np.array([pearsonr(orig, den)[0] 
                         for orig, den in zip(original, denoised)])
    metrics['mean_row_correlation'] = np.mean(row_corrs)
    
    # Column-wise correlations (across samples)
    col_corrs = np.array([pearsonr(original[:, i], denoised[:, i])[0] 
                         for i in range(original.shape[1])])
    metrics['mean_column_correlation'] = np.mean(col_corrs)
    
    # 4. Signal-to-Noise Ratio
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / noise_power)
    
    original_noise = noisy - original
    residual_noise = denoised - original
    metrics['original_snr'] = calculate_snr(original, original_noise)
    metrics['denoised_snr'] = calculate_snr(original, residual_noise)
    metrics['snr_improvement'] = metrics['denoised_snr'] - metrics['original_snr']
    
    # 5. Silhouette Coefficients
    # For original data
    metrics['silhouette_original'] = silhouette_score(original, labels)
    # For noisy data
    metrics['silhouette_noisy'] = silhouette_score(noisy, labels)
    # For denoised data
    metrics['silhouette_denoised'] = silhouette_score(denoised, labels)
    # Improvement in silhouette score
    metrics['silhouette_improvement_vs_noisy'] = metrics['silhouette_denoised'] - metrics['silhouette_noisy']
    metrics['silhouette_improvement_vs_original'] = metrics['silhouette_denoised'] - metrics['silhouette_original']
    
    return metrics