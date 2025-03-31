# algorithms/ednc/coding.py
"""
Coding parameter optimization for the E-EDNC algorithm.

This module determines optimal network coding parameters (degree and scheme)
using convex optimization based on entropy and network conditions.
"""

import numpy as np

# Import optimization tools with fallback
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def optimize_coding_parameters(algorithm, entropy, network_conditions):
    """
    Determine optimal coding parameters using convex optimization.
    
    This method reformulates the coding efficiency optimization as a convex problem
    and determines the optimal coding degree and scheme.
    """
    if not SCIPY_AVAILABLE:
        # Fallback if scipy not available
        return heuristic_coding_parameters(algorithm, entropy, network_conditions)
    
    # Define coding efficiency function
    def coding_efficiency(params):
        dt, scheme_idx = params
        dt = int(round(dt))  # Coding degree must be integer
        
        # Ensure dt is within range
        dt = max(algorithm.coding_degrees_range[0], 
                min(algorithm.coding_degrees_range[1], dt))
        
        # Select coding scheme based on scheme_idx
        scheme_idx = int(round(scheme_idx))
        scheme_idx = max(0, min(len(algorithm.coding_schemes) - 1, scheme_idx))
        coding_scheme = algorithm.coding_schemes[scheme_idx]
        
        # Coding scheme efficiency parameters
        lambda_ct = get_scheme_lambda(coding_scheme)
        
        # Mutual information
        mutual_information = entropy * (1 - np.exp(-lambda_ct * dt))
        
        # Overhead based on scheme and degree
        overhead_factors = get_scheme_overhead(coding_scheme)
        
        # Bandwidth consumption with overhead
        bandwidth = dt * (np.log2(algorithm.data_alphabet_size) + overhead_factors * dt)
        
        # Coding efficiency (objective to maximize)
        efficiency = mutual_information / bandwidth
        
        # Return negative efficiency for minimization
        return -efficiency
    
    # Network condition impacts
    condition_impacts = get_network_condition_impacts(network_conditions)
    
    # Latency constraint function
    def latency_constraint(params):
        dt, scheme_idx = params
        dt = int(round(dt))
        scheme_idx = int(round(scheme_idx))
        scheme_idx = max(0, min(len(algorithm.coding_schemes) - 1, scheme_idx))
        coding_scheme = algorithm.coding_schemes[scheme_idx]
        
        # Processing time factors by scheme
        processing_factors = get_scheme_processing_factor(coding_scheme)
        
        # Calculate latency
        latency = dt * condition_impacts["latency_factor"] * processing_factors
        
        # Return constraint satisfaction: latency <= max_latency
        margin = getattr(algorithm, 'latency_margin', 1.0)
        return algorithm.max_latency * margin - latency
    
    # Reliability benefit function (not a constraint, but influences objective)
    def reliability_benefit(params):
        dt, scheme_idx = params
        dt = int(round(dt))
        scheme_idx = int(round(scheme_idx))
        scheme_idx = max(0, min(len(algorithm.coding_schemes) - 1, scheme_idx))
        coding_scheme = algorithm.coding_schemes[scheme_idx]
        
        reliability_factors = get_scheme_reliability(coding_scheme)
        
        # Higher coding degree increases reliability
        reliability = (reliability_factors * 
                      condition_impacts["reliability_factor"] * 
                      (1 - np.exp(-0.1 * dt)))
        
        return reliability
    
    # Combined objective function with reliability consideration
    def combined_objective(params):
        efficiency_factor = coding_efficiency(params)
        reliability_weight = getattr(algorithm, 'reliability_weight', 0.3)
        reliability_factor = -reliability_weight * reliability_benefit(params)
        
        return efficiency_factor + reliability_factor
    
    # Initial guess based on entropy level
    initial_dt = algorithm.coding_degrees_range[0] + entropy * (
        algorithm.coding_degrees_range[1] - algorithm.coding_degrees_range[0])
    
    # Scheme selection initial guess
    if entropy < 0.3:
        initial_scheme = 0  # Simple
    elif entropy < 0.7:
        initial_scheme = 1  # Fountain
    else:
        initial_scheme = 2  # RLNC
    
    # Bounds for optimization
    bounds = [(algorithm.coding_degrees_range[0], algorithm.coding_degrees_range[1]), 
              (0, len(algorithm.coding_schemes) - 1)]
    
    # Run optimization
    try:
        result = minimize(combined_objective, 
                        [initial_dt, initial_scheme],
                        bounds=bounds,
                        constraints=[{'type': 'ineq', 'fun': latency_constraint}],
                        method='SLSQP')
        
        # Extract results
        optimal_dt = int(round(result.x[0]))
        optimal_scheme_idx = int(round(result.x[1]))
        
        # Ensure results are within allowed ranges
        optimal_dt = max(algorithm.coding_degrees_range[0], 
                        min(algorithm.coding_degrees_range[1], optimal_dt))
        optimal_scheme_idx = max(0, min(len(algorithm.coding_schemes) - 1, 
                                        optimal_scheme_idx))
        
        optimal_scheme = algorithm.coding_schemes[optimal_scheme_idx]
        
        return {
            "coding_degree": optimal_dt,
            "coding_scheme": optimal_scheme
        }
    
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        # Fallback to heuristic algorithm
        return heuristic_coding_parameters(algorithm, entropy, network_conditions)

def heuristic_coding_parameters(algorithm, entropy, network_conditions):
    """
    Fallback heuristic algorithm to determine coding parameters.
    """
    # Determine coding degree based on entropy
    min_degree, max_degree = algorithm.coding_degrees_range
    coding_degree = min_degree + int((max_degree - min_degree) * entropy)
    
    # Select coding scheme based on entropy and network conditions
    if entropy < 0.3:
        coding_scheme = "Simple"
    elif entropy < 0.7:
        if network_conditions == "congested":
            coding_scheme = "RLNC"  # More robust for congestion
        else:
            coding_scheme = "Fountain"
    else:
        coding_scheme = "RLNC"
        
    return {
        "coding_degree": coding_degree,
        "coding_scheme": coding_scheme
    }

# Helper functions

def get_scheme_lambda(scheme):
    """Get lambda parameter for coding scheme."""
    scheme_params = {
        "Simple": 0.8,
        "Fountain": 1.0,
        "RLNC": 1.2
    }
    return scheme_params.get(scheme, 1.0)

def get_scheme_overhead(scheme):
    """Get overhead factor for coding scheme."""
    overhead_factors = {
        "Simple": 0.01,
        "Fountain": 0.03,
        "RLNC": 0.05
    }
    return overhead_factors.get(scheme, 0.03)

def get_scheme_processing_factor(scheme):
    """Get processing time factor for coding scheme."""
    processing_factors = {
        "Simple": 0.5,
        "Fountain": 1.0,
        "RLNC": 1.5
    }
    return processing_factors.get(scheme, 1.0)

def get_scheme_reliability(scheme):
    """Get base reliability for coding scheme."""
    reliability_factors = {
        "Simple": 0.85,
        "Fountain": 0.92,
        "RLNC": 0.98
    }
    return reliability_factors.get(scheme, 0.9)

def get_network_condition_impacts(condition):
    """Get impact factors for network condition."""
    condition_impacts = {
        "normal": {"latency_factor": 1.0, "reliability_factor": 1.0},
        "congested": {"latency_factor": 2.0, "reliability_factor": 0.8},
        "interference": {"latency_factor": 1.5, "reliability_factor": 0.7}
    }
    return condition_impacts.get(condition, 
                               {"latency_factor": 1.0, "reliability_factor": 1.0})
