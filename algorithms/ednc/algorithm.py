# algorithms/ednc/algorithm.py
"""
Main implementation of the Entropy-Driven Network Coding algorithm.

This module provides the main algorithm class that integrates all components:
entropy estimation, entropy prediction, coding parameter optimization,
hybrid evolutionary-reinforcement learning scheduling, and feedback processing.
"""

import numpy as np
from collections import deque, defaultdict

from ..base import NetworkCodingAlgorithm
from . import entropy, coding, scheduling, feedback

class EDNCAlgorithm(NetworkCodingAlgorithm):
    """Implementation of the Enhanced Entropy-Driven Network Coding (E-EDNC) algorithm."""
    
    def initialize(self, params):
        """Initialize algorithm with parameters."""
        self.params = params
        
        # Extract parameters with defaults
        self.sliding_window_size = params.get("sliding_window_size", 100)
        self.data_alphabet_size = params.get("data_alphabet_size", 256)
        self.tensor_dimensionality = params.get("tensor_dimensionality", 3)
        self.coding_degrees_range = params.get("coding_degrees_range", (2, 10))
        self.coding_schemes = params.get("coding_schemes", ["RLNC", "Fountain", "Simple"])
        self.max_latency = params.get("max_latency", 100)
        self.total_bandwidth = params.get("total_bandwidth", 1e9)
        self.total_energy = params.get("total_energy", 100)
        
        # Initialize data structures
        self.data_windows = {}  # For tensor-based entropy estimation
        self.entropy_history = defaultdict(list)  # device_id -> list of entropy values
        self.entropy_predictions = []
        self.entropy_actuals = []
        
        # AREP model parameters
        self.arep_model_order = 5
        self.ar_coefficients = np.array([0.6, 0.25, 0.1, 0.04, 0.01])
        self.prediction_noise_std = 0.05
        
        # Optimization parameters
        self.latency_margin = 1.0
        self.reliability_weight = 0.3
        
        # Initialize HE-RL parameters
        scheduling.initialize_policies(self)
        
        # Feedback processing parameters
        self.feedback_history = []
        self.entropy_adjustment_factors = {
            "latency": 0.1,
            "reliability": 0.1,
            "congestion": 0.1
        }
        
        print("E-EDNC algorithm initialized with parameters:")
        print(f"- Sliding window size: {self.sliding_window_size}")
        print(f"- Data alphabet size: {self.data_alphabet_size}")
        print(f"- Tensor dimensionality: {self.tensor_dimensionality}")
        print(f"- Coding degrees range: {self.coding_degrees_range}")
        print(f"- Coding schemes: {self.coding_schemes}")
    
    def estimate_entropy(self, data, sliding_window):
        """Entropy estimation using tensor-based approach."""
        estimated_entropy = entropy.estimate_tensor_entropy(self, data, sliding_window)
        
        # Store actual entropy for model validation
        if "entropy" in data:
            device_id = data.get("device_id", "unknown")
            self.entropy_actuals.append(data["entropy"])
            self.entropy_history[device_id].append(data["entropy"])
        
        return estimated_entropy
    
    def predict_entropy(self, entropy_history):
        """Predict future entropy using AREP model."""
        predicted_entropy = entropy.predict_entropy(self, entropy_history)
        
        # Store prediction for model validation
        self.entropy_predictions.append(predicted_entropy)
        
        return predicted_entropy
    
    def determine_coding_parameters(self, entropy, network_conditions):
        """Determine optimal coding parameters using convex optimization."""
        return coding.optimize_coding_parameters(self, entropy, network_conditions)
    
    def schedule_packets(self, packets, constraints):
        """Schedule packets using the HE-RL algorithm."""
        return scheduling.he_rl_scheduling(self, packets, constraints)
    
    def process_feedback(self, feedback_data):
        """Process feedback to adapt algorithm parameters."""
        if feedback_data:
            self.feedback_history.append(feedback_data)
            return feedback.process_feedback(self, feedback_data)
