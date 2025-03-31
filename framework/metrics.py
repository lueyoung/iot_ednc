# framework/metrics.py
"""
Performance metrics calculation for the IoT simulation framework.

This module provides functions for calculating network performance metrics
and generating feedback for algorithm adaptation.
"""

import math
import numpy as np

def calculate_network_metrics(packet, coding_params, config, fog_nodes):
    """
    Calculate network performance metrics for a packet.
    
    Args:
        packet (dict): Packet data
        coding_params (dict): Coding parameters (degree and scheme)
        config (dict): Simulation configuration
        fog_nodes (list): List of fog nodes
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Get fog node
    fog_node = next(f for f in fog_nodes if f["fog_id"] == packet["fog_node"])
    
    # Calculate base latency based on distance
    device_x = packet.get("zone_x", 0)
    device_y = packet.get("zone_y", 0)
    distance = math.sqrt((device_x - fog_node["x"])**2 + (device_y - fog_node["y"])**2)
    base_latency = distance * 0.05  # 0.05 ms per distance unit
    
    # Get network condition and coding scheme factors
    condition = packet["network_condition"]
    condition_factors = config["network_conditions"][condition]
    
    coding_scheme = coding_params["coding_scheme"]
    coding_degree = coding_params["coding_degree"]
    
    # Coding scheme factors
    coding_factors = {
        "Simple": {"latency": 1.0, "reliability": 0.9, "energy": 0.8, "lambda": 0.8},
        "Fountain": {"latency": 1.2, "reliability": 0.95, "energy": 1.0, "lambda": 1.0},
        "RLNC": {"latency": 1.3, "reliability": 0.98, "energy": 1.2, "lambda": 1.2}
    }
    
    scheme_factors = coding_factors.get(coding_scheme, coding_factors["Fountain"])
    
    # Calculate metrics
    packet_size = packet["packet_size"]
    
    # Latency calculation
    latency = base_latency * condition_factors["latency_factor"] * scheme_factors["latency"]
    latency += (packet_size / 1024) * coding_degree * 0.5  # Additional latency based on size and coding
    
    # Reliability calculation
    reliability = condition_factors["reliability_factor"] * scheme_factors["reliability"]
    reliability += (coding_degree - 2) * 0.01  # Higher coding degree increases reliability
    reliability = min(0.999, reliability)  # Cap at 99.9%
    
    # Energy consumption calculation
    energy = (packet_size / 128) * scheme_factors["energy"]
    energy += (coding_degree - 2) * 0.02  # Higher coding degree increases energy consumption
    
    # Bandwidth usage calculation
    bandwidth_usage = packet_size * coding_degree * 8  # bits
    
    # Mutual information calculation
    entropy = packet["entropy"]
    lambda_factor = scheme_factors["lambda"]
    mutual_information = entropy * (1 - math.exp(-lambda_factor * coding_degree))
    
    # Throughput calculation (Mbps)
    throughput = mutual_information / (bandwidth_usage / 1e6)
    
    return {
        "latency": latency,
        "reliability": reliability,
        "energy": energy,
        "bandwidth_usage": bandwidth_usage,
        "mutual_information": mutual_information,
        "throughput": throughput
    }

def generate_feedback(time_step, processed_packets, window_size=1000):
    """
    Generate feedback metrics for algorithm adjustment.
    
    Args:
        time_step (int): Current simulation time step
        processed_packets (list): List of processed packets
        window_size (int): Number of recent packets to consider
        
    Returns:
        dict: Feedback metrics
    """
    if not processed_packets:
        return None
    
    # Consider only recent packets
    recent_packets = processed_packets[-min(len(processed_packets), window_size):]
    
    # Calculate average metrics
    avg_latency = np.mean([p["metrics"]["latency"] for p in recent_packets])
    avg_reliability = np.mean([p["metrics"]["reliability"] for p in recent_packets])
    avg_energy = np.mean([p["metrics"]["energy"] for p in recent_packets])
    avg_bandwidth = np.mean([p["metrics"]["bandwidth_usage"] for p in recent_packets])
    avg_throughput = np.mean([p["metrics"]["throughput"] for p in recent_packets])
    
    # Calculate congestion level
    congestion_level = sum(1 for p in recent_packets 
                         if p["packet"]["network_condition"] == "congested") / len(recent_packets)
    
    # Calculate coding efficiency
    avg_mutual_info = np.mean([p["metrics"]["mutual_information"] for p in recent_packets])
    coding_efficiency = avg_mutual_info / (avg_bandwidth / 8)  # bits to bytes
    
    return {
        "time_step": time_step,
        "avg_latency": avg_latency,
        "avg_reliability": avg_reliability,
        "avg_energy": avg_energy,
        "avg_bandwidth": avg_bandwidth,
        "avg_throughput": avg_throughput,
        "congestion_level": congestion_level,
        "coding_efficiency": coding_efficiency,
        "performance_score": calculate_performance_score(
            avg_reliability, avg_latency, avg_energy)
    }

def calculate_performance_score(reliability, latency, energy):
    """
    Calculate overall performance score.
    
    Args:
        reliability (float): Reliability metric (higher is better)
        latency (float): Latency metric (lower is better)
        energy (float): Energy consumption metric (lower is better)
        
    Returns:
        float: Combined performance score
    """
    # Normalize metrics
    normalized_latency = min(1.0, latency / 100)  # Normalize to [0,1]
    normalized_energy = min(1.0, energy / 0.1)    # Normalize to [0,1]
    
    # Weighted sum (higher is better)
    score = (reliability * 0.4) - (normalized_latency * 0.3) - (normalized_energy * 0.3)
    
    return score
