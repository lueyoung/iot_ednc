# algorithms/ednc/feedback.py
"""
Feedback processing mechanism for the E-EDNC algorithm.

This module handles the processing of performance feedback to adjust
algorithm parameters dynamically based on network conditions.
"""

import numpy as np
from . import scheduling

def process_feedback(algorithm, feedback):
    """
    Process feedback to adapt algorithm parameters based on 
    network performance metrics.
    """
    if not feedback:
        return
    
    # Store feedback
    if not hasattr(algorithm, 'feedback_history'):
        algorithm.feedback_history = []
    algorithm.feedback_history.append(feedback)
    
    # Keep only recent history for efficiency
    max_history = 10
    if len(algorithm.feedback_history) > max_history:
        algorithm.feedback_history = algorithm.feedback_history[-max_history:]
    
    # Skip if not enough history
    if len(algorithm.feedback_history) < 2:
        return
    
    # Update parameters based on feedback trends
    update_entropy_adjustment(algorithm, feedback)
    update_ar_model_based_on_feedback(algorithm, feedback)
    update_coding_parameters_based_on_feedback(algorithm, feedback)
    
    # Update HE-RL algorithm parameters
    if hasattr(algorithm, 'current_policy_idx'):
        scheduling.reinforcement_learning_update(algorithm, algorithm.current_policy_idx, feedback)

def update_entropy_adjustment(algorithm, feedback):
    """Update entropy adjustment factors based on performance feedback."""
    # Initialize adjustment parameters if needed
    if not hasattr(algorithm, 'entropy_adjustment_factors'):
        algorithm.entropy_adjustment_factors = {
            "latency": 0.1,
            "reliability": 0.1,
            "congestion": 0.1
        }
    
    # Calculate adjustment direction based on performance
    latency_trend = 0
    reliability_trend = 0
    congestion_trend = 0
    
    # Compare with previous feedback
    if len(algorithm.feedback_history) >= 2:
        prev = algorithm.feedback_history[-2]
        current = feedback
        
        # Latency trend (negative is better)
        if current["avg_latency"] > prev["avg_latency"]:
            latency_trend = -0.01  # Increase entropy threshold to reduce latency
        else:
            latency_trend = 0.005
        
        # Reliability trend (positive is better)
        if current["avg_reliability"] < prev["avg_reliability"]:
            reliability_trend = -0.01  # Decrease entropy threshold to improve reliability
        else:
            reliability_trend = 0.005
        
        # Congestion trend (negative is better)
        if current["congestion_level"] > prev["congestion_level"]:
            congestion_trend = -0.01  # Adjust for congestion
        else:
            congestion_trend = 0.005
    
    # Update factors with small steps
    algorithm.entropy_adjustment_factors["latency"] += latency_trend
    algorithm.entropy_adjustment_factors["reliability"] += reliability_trend
    algorithm.entropy_adjustment_factors["congestion"] += congestion_trend
    
    # Keep factors in reasonable range
    for key in algorithm.entropy_adjustment_factors:
        algorithm.entropy_adjustment_factors[key] = max(-0.2, min(0.2, algorithm.entropy_adjustment_factors[key]))

def update_ar_model_based_on_feedback(algorithm, feedback):
    """Update AR model parameters based on network feedback."""
    # Skip if no AR model initialized
    if not hasattr(algorithm, 'ar_coefficients') or not hasattr(algorithm, 'prediction_noise_std'):
        return
    
    # Adjust noise parameter based on congestion
    congestion_level = feedback.get("congestion_level", 0)
    
    # Increase noise parameter for more volatile environments
    if congestion_level > 0.5:
        algorithm.prediction_noise_std = min(0.1, algorithm.prediction_noise_std * 1.05)
    else:
        algorithm.prediction_noise_std = max(0.01, algorithm.prediction_noise_std * 0.95)
    
    # If prediction performance is poor, trigger retraining
    if hasattr(algorithm, 'entropy_predictions') and hasattr(algorithm, 'entropy_actuals'):
        if len(algorithm.entropy_predictions) > 10 and len(algorithm.entropy_actuals) > 10:
            pred_error = np.mean([abs(p - a) for p, a in zip(
                algorithm.entropy_predictions[-10:], 
                algorithm.entropy_actuals[-10:])])
            
            if pred_error > 0.15:  # High prediction error
                algorithm.ar_retraining_needed = True

def update_coding_parameters_based_on_feedback(algorithm, feedback):
    """Update coding parameter optimization constraints based on feedback."""
    # Adjust latency constraint based on current performance
    current_latency = feedback.get("avg_latency", 0)
    
    # If we're well below latency constraint, we can relax it
    if hasattr(algorithm, 'max_latency') and current_latency < 0.7 * algorithm.max_latency:
        # Allow more complex coding schemes
        if not hasattr(algorithm, 'latency_margin'):
            algorithm.latency_margin = 1.1
        else:
            algorithm.latency_margin = min(1.2, algorithm.latency_margin * 1.05)
    
    # If we're close to latency constraint, tighten it
    elif hasattr(algorithm, 'max_latency') and current_latency > 0.9 * algorithm.max_latency:
        # Force simpler coding schemes
        if not hasattr(algorithm, 'latency_margin'):
            algorithm.latency_margin = 0.9
        else:
            algorithm.latency_margin = max(0.8, algorithm.latency_margin * 0.95)
    
    # Adjust reliability weight based on current reliability
    current_reliability = feedback.get("avg_reliability", 0)
    
    if current_reliability < 0.9:
        # Increase weight of reliability in optimization
        if not hasattr(algorithm, 'reliability_weight'):
            algorithm.reliability_weight = 0.35
        else:
            algorithm.reliability_weight = min(0.5, algorithm.reliability_weight * 1.1)
    else:
        # Can focus more on efficiency
        if not hasattr(algorithm, 'reliability_weight'):
            algorithm.reliability_weight = 0.25
        else:
            algorithm.reliability_weight = max(0.1, algorithm.reliability_weight * 0.95)

def calculate_performance_score(feedback):
    """
    Calculate a single performance score from multiple metrics.
    Used for policy evaluation.
    """
    if not feedback:
        return 0
    
    # Weighted combination of metrics
    reliability = feedback.get("avg_reliability", 0)
    latency = feedback.get("avg_latency", 0)
    energy = feedback.get("avg_energy", 0)
    
    # Normalize and combine
    normalized_latency = min(1.0, latency / 100)  # Normalize to [0,1]
    normalized_energy = min(1.0, energy / 0.1)    # Normalize to [0,1]
    
    # Higher is better, so we subtract latency and energy
    score = (reliability * 0.4) - (normalized_latency * 0.3) - (normalized_energy * 0.3)
    
    return score
