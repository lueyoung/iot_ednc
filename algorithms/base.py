# algorithms/base.py
"""
Abstract base class for network coding algorithms.

This module defines the interface that all network coding algorithms
must implement to be compatible with the simulation framework.
"""

from abc import ABC, abstractmethod

class NetworkCodingAlgorithm(ABC):
    """Base interface for network coding algorithms."""
    
    @abstractmethod
    def initialize(self, params):
        """
        Initialize algorithm with necessary parameters.
        
        Args:
            params (dict): Configuration parameters for the algorithm
        """
        pass
    
    @abstractmethod
    def estimate_entropy(self, data, sliding_window):
        """
        Estimate entropy of the given data using sliding window approach.
        
        Args:
            data (dict): Packet data containing raw content and metadata
            sliding_window (int): Size of sliding window for entropy estimation
            
        Returns:
            float: Estimated entropy value between 0 and 1
        """
        pass
    
    @abstractmethod
    def predict_entropy(self, entropy_history):
        """
        Predict future entropy values based on history.
        
        Args:
            entropy_history (list): Historical entropy values
            
        Returns:
            float: Predicted entropy value for the next time step
        """
        pass
    
    @abstractmethod
    def determine_coding_parameters(self, entropy, network_conditions):
        """
        Determine optimal coding parameters based on entropy and network conditions.
        
        Args:
            entropy (float): Current entropy value
            network_conditions (str): Current network condition (e.g., "normal", "congested")
            
        Returns:
            dict: Dictionary containing coding parameters like degree and scheme
        """
        pass
    
    @abstractmethod
    def schedule_packets(self, packets, constraints):
        """
        Schedule packets based on priority and network constraints.
        
        Args:
            packets (list): List of packets to be scheduled
            constraints (dict): Resource constraints (bandwidth, energy, etc.)
            
        Returns:
            list: Scheduled packets in order of transmission
        """
        pass
    
    @abstractmethod
    def process_feedback(self, feedback):
        """
        Process feedback to adjust algorithm parameters.
        
        Args:
            feedback (dict): Performance metrics and network state information
            
        Returns:
            None
        """
        pass
