# algorithms/default.py
"""
Default implementation of the network coding algorithm.

This module provides a simple default implementation of the network coding
algorithm interface to be used when a more advanced algorithm is not available
or for baseline comparison purposes.
"""

import random
import numpy as np
from collections import deque, Counter
from .base import NetworkCodingAlgorithm

class DefaultAlgorithm(NetworkCodingAlgorithm):
    """Default implementation with basic functionality."""
    
    def initialize(self, params):
        """Initialize algorithm with parameters."""
        self.params = params
        
        # Extract key parameters with defaults
        self.sliding_window_size = params.get("sliding_window_size", 100)
        self.data_alphabet_size = params.get("data_alphabet_size", 256)
        self.tensor_dimensionality = params.get("tensor_dimensionality", 3)
        self.coding_degrees_range = params.get("coding_degrees_range", (2, 10))
        self.coding_schemes = params.get("coding_schemes", ["RLNC", "Fountain", "Simple"])
        self.max_latency = params.get("max_latency", 100)
        
        # Initialize data structures
        self.data_windows = {}
        self.entropy_history = {}
        self.feedback_count = 0
        
        print("Default algorithm initialized with parameters:")
        print(f"- Sliding window size: {self.sliding_window_size}")
        print(f"- Coding degrees range: {self.coding_degrees_range}")
        print(f"- Coding schemes: {self.coding_schemes}")
    
    def estimate_entropy(self, data, sliding_window):
        """
        Estimate entropy using Shannon's formula.
        
        This is a basic implementation using traditional entropy calculation
        rather than the tensor-based approach used in E-EDNC.
        """
        # Use the provided entropy if available
        if "entropy" in data:
            return data["entropy"]
        
        # Extract raw content
        if "raw_content" in data:
            content = data["raw_content"]
            if isinstance(content, list) and len(content) > 0:
                return self._calculate_shannon_entropy(content)
        
        # Fallback to default medium entropy
        return 0.5
    
    def predict_entropy(self, entropy_history):
        """
        Predict future entropy using simple moving average.
        
        This is a basic prediction approach, not the auto-regressive
        model used in E-EDNC.
        """
        if not entropy_history:
            return 0.5
        
        # Use moving average of last few values
        window_size = min(5, len(entropy_history))
        if window_size > 0:
            return sum(entropy_history[-window_size:]) / window_size
        
        return entropy_history[-1] if entropy_history else 0.5
    
    def determine_coding_parameters(self, entropy, network_conditions):
        """
        Determine coding parameters using heuristic rules.
        
        This is a simplified approach using predefined rules instead of
        the convex optimization used in E-EDNC.
        """
        # Calculate coding degree based on entropy
        min_degree, max_degree = self.coding_degrees_range
        coding_degree = min_degree + round((max_degree - min_degree) * entropy)
        
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
    
    def schedule_packets(self, packets, constraints):
        """
        Schedule packets using a simple priority-based approach.
        
        This is a basic scheduling approach instead of the HE-RL
        algorithm used in E-EDNC.
        """
        # Calculate priority for each packet: higher entropy = higher priority
        priorities = []
        for packet in packets:
            # Priority based on entropy and packet size
            entropy = packet.get("entropy", 0.5)
            packet_size = packet.get("packet_size", 1024)
            
            # Simple priority formula
            priority = entropy - (packet_size / 10000)  # Size penalty
            
            # Network condition adjustment
            if packet.get("network_condition") == "congested":
                priority *= 0.8  # Lower priority in congested networks
            
            priorities.append((packet, priority))
        
        # Sort by priority (descending)
        sorted_packets = [p[0] for p in sorted(priorities, key=lambda x: x[1], reverse=True)]
        
        # Apply constraints
        scheduled_packets = []
        total_size = 0
        total_energy = 0
        
        for packet in sorted_packets:
            packet_size = packet.get("packet_size", 1024)
            # Estimate energy based on size
            estimated_energy = packet_size / 1024  
            
            if (total_size + packet_size <= constraints.get("bandwidth", float('inf')) and
                total_energy + estimated_energy <= constraints.get("energy", float('inf'))):
                scheduled_packets.append(packet)
                total_size += packet_size
                total_energy += estimated_energy
        
        return scheduled_packets
    
    def process_feedback(self, feedback):
        """
        Process feedback to adjust algorithm parameters.
        
        This is a minimal implementation that simply counts feedback events
        but doesn't adjust parameters like E-EDNC does.
        """
        if feedback:
            self.feedback_count += 1
            
            # Log feedback every 10 events
            if self.feedback_count % 10 == 0:
                print(f"Feedback event {self.feedback_count}:")
                print(f"- Average latency: {feedback.get('avg_latency', 'N/A')}")
                print(f"- Average reliability: {feedback.get('avg_reliability', 'N/A')}")
                print(f"- Congestion level: {feedback.get('congestion_level', 'N/A')}")
    
    def _calculate_shannon_entropy(self, data):
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
