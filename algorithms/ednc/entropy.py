# algorithms/ednc/entropy.py
"""
Entropy estimation and prediction modules for the E-EDNC algorithm.

This module implements tensor-based entropy estimation and
auto-regressive entropy prediction (AREP) models.
"""

import numpy as np
from collections import deque, Counter
from scipy.stats import entropy as scipy_entropy

# Tensor operations (with safe imports)
try:
    import tensorly as tl
    from tensorly.decomposition import tucker
    TENSORLY_AVAILABLE = True
except ImportError:
    TENSORLY_AVAILABLE = False

def estimate_tensor_entropy(algorithm, data, sliding_window):
    """
    Estimate entropy using tensor-based approach.
    
    This method constructs a tensor from sliding window data, performs HOSVD,
    and computes entropy based on tensor decomposition.
    """
    # Extract raw content or use provided data
    if "raw_content" in data:
        content = data["raw_content"]
    else:
        # Use the entropy value directly if raw content isn't available
        return data["entropy"]
    
    # Get device ID or use a default
    device_id = data.get("device_id", "unknown")
    
    # Create window of recent data or use existing window
    if device_id not in algorithm.data_windows:
        algorithm.data_windows[device_id] = deque(maxlen=algorithm.sliding_window_size)
    
    window_data = algorithm.data_windows[device_id]
    
    # Add new content to window
    if isinstance(content, list) and len(content) > 0:
        window_data.append(content)
    
    # Check if we have enough data for tensor analysis
    if len(window_data) < 3 or not TENSORLY_AVAILABLE:
        # Fall back to traditional entropy calculation
        if isinstance(content, list) and len(content) > 0:
            return calculate_shannon_entropy(content)
        return data["entropy"]
    
    try:
        # Construct tensor with dimensions: [window_size × alphabet_size × feature_dim]
        tensor_shape = (min(len(window_data), sliding_window), 
                        algorithm.data_alphabet_size, 
                        min(algorithm.tensor_dimensionality, 3))
        
        # Initialize tensor with zeros
        data_tensor = np.zeros(tensor_shape)
        
        # Fill tensor with data
        for i, sample in enumerate(list(window_data)[-tensor_shape[0]:]):
            if isinstance(sample, list):
                for j, value in enumerate(sample[:min(len(sample), tensor_shape[2])]):
                    if 0 <= value < algorithm.data_alphabet_size:
                        data_tensor[i, value, j % tensor_shape[2]] += 1
        
        # Normalize tensor
        data_tensor = data_tensor / (np.sum(data_tensor) + 1e-10)
        
        # Perform Tucker decomposition (HOSVD)
        core, factors = tucker(data_tensor, rank=[min(d, 5) for d in tensor_shape])
        
        # Compute entropy based on core tensor
        core_flat = core.flatten()
        core_flat = np.abs(core_flat) / (np.sum(np.abs(core_flat)) + 1e-10)
        tensor_entropy = scipy_entropy(core_flat, base=2)
        
        # Scale entropy to [0,1]
        max_possible_entropy = np.log2(np.prod(core.shape))
        normalized_entropy = min(1.0, tensor_entropy / max_possible_entropy)
        
        # Apply regularization using original entropy
        original_entropy = data["entropy"]
        alpha = 0.7  # Weight for tensor-based estimate
        combined_entropy = alpha * normalized_entropy + (1 - alpha) * original_entropy
        
        return combined_entropy
        
    except Exception as e:
        # Fall back to original entropy on decomposition failure
        print(f"Tensor entropy estimation failed: {str(e)}")
        return data["entropy"]

def calculate_shannon_entropy(data):
    """Calculate Shannon entropy of a data sequence."""
    if not data:
        return 0
    
    # Count frequency of each value
    counter = Counter(data)
    
    # Calculate probabilities
    n = len(data)
    probabilities = [count / n for count in counter.values()]
    
    # Calculate entropy
    raw_entropy = -sum(p * np.log2(p) for p in probabilities)
    
    # Normalize to [0,1] using maximum possible entropy
    max_entropy = np.log2(len(counter))
    if max_entropy == 0:
        return 0
    
    return min(1.0, raw_entropy / max_entropy)

def predict_entropy(algorithm, entropy_history):
    """
    Predict future entropy using Auto-Regressive Entropy Prediction (AREP) model.
    
    The model predicts future entropy values based on historical entropy observations
    using an autoregressive model of order k.
    """
    if not entropy_history or len(entropy_history) < 2:
        return entropy_history[-1] if entropy_history else 0.5
    
    # Determine model order k based on available history
    k = min(algorithm.arep_model_order, len(entropy_history) - 1)
    
    # If coefficients aren't initialized or need updating
    if len(entropy_history) >= k + 10:
        update_ar_coefficients(algorithm, entropy_history, k)
    
    # Apply AR model to predict next entropy value
    prediction = 0
    for i in range(k):
        idx = -(i+1)
        if idx >= -len(entropy_history):
            prediction += algorithm.ar_coefficients[i] * entropy_history[idx]
    
    # Add white Gaussian noise (optional, for more realistic prediction)
    if hasattr(algorithm, 'prediction_noise_std'):
        prediction += np.random.normal(0, algorithm.prediction_noise_std)
    
    # Ensure entropy is within valid range [0,1]
    prediction = max(0, min(1, prediction))
    
    return prediction

def update_ar_coefficients(algorithm, entropy_history, k):
    """
    Update AR model coefficients using Yule-Walker equations.
    """
    # For stability, ensure we have enough data points
    if len(entropy_history) < k + 10:
        # Keep existing coefficients
        return
    
    try:
        # Use statsmodels for AR model estimation if available
        try:
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(entropy_history, lags=k)
            model_fit = model.fit()
            algorithm.ar_coefficients = model_fit.params[1:]  # Skip the intercept
            algorithm.prediction_noise_std = np.sqrt(model_fit.sigma2)
            return
        except ImportError:
            pass  # Proceed with manual implementation
        
        # Manual implementation of Yule-Walker method
        r = np.zeros(k+1)
        for i in range(k+1):
            r[i] = np.sum([(entropy_history[j-i] - np.mean(entropy_history)) * 
                          (entropy_history[j] - np.mean(entropy_history)) 
                          for j in range(i, len(entropy_history))]) / (len(entropy_history) - i)
        
        # Form Toeplitz matrix
        R = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                R[i, j] = r[abs(i-j)]
        
        # Solve system of equations with regularization
        algorithm.ar_coefficients = np.linalg.solve(R + 1e-6 * np.eye(k), r[1:k+1])
        
        # Ensure stability by constraining sum of coefficients
        if np.sum(np.abs(algorithm.ar_coefficients)) >= 1:
            algorithm.ar_coefficients = algorithm.ar_coefficients / (np.sum(np.abs(algorithm.ar_coefficients)) + 0.01)
        
        # Estimate noise variance
        residuals = np.zeros(len(entropy_history) - k)
        for i in range(k, len(entropy_history)):
            pred = np.sum([algorithm.ar_coefficients[j] * entropy_history[i-j-1] for j in range(k)])
            residuals[i-k] = entropy_history[i] - pred
            
        algorithm.prediction_noise_std = np.std(residuals)
    
    except Exception as e:
        print(f"AR coefficient update failed: {str(e)}")
        # Keep existing coefficients on failure
