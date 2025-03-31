# framework/entities.py
"""
Simulation entity definitions for the IoT framework.

This module provides functions for creating and managing IoT devices
and fog nodes in the simulation environment.
"""

import random
import math
import numpy as np
from scipy.stats import zipf

def initialize_devices(num_devices, device_types):
    """
    Initialize IoT devices with realistic characteristics.
    
    Args:
        num_devices (int): Number of devices to create
        device_types (dict): Device type characteristics
        
    Returns:
        list: List of device objects
    """
    devices = []
    
    # Define device distribution by type
    device_distribution = {
        "temperature_sensor": 0.25,
        "humidity_sensor": 0.15,
        "motion_detector": 0.20,
        "camera": 0.05,
        "smart_meter": 0.20,
        "health_monitor": 0.15
    }
    
    device_type_counts = {}
    for device_type, proportion in device_distribution.items():
        device_type_counts[device_type] = int(num_devices * proportion)
        
    # Adjust to ensure we have exactly the required number of devices
    remainder = num_devices - sum(device_type_counts.values())
    device_type_counts["temperature_sensor"] += remainder
    
    device_id = 0
    for device_type, count in device_type_counts.items():
        for _ in range(count):
            # Assign temporal pattern weights
            temporal_weights = {
                "daily": random.uniform(0.3, 1.0) if random.random() < 0.8 else 0,
                "weekly": random.uniform(0.3, 1.0) if random.random() < 0.6 else 0,
                "random_spike": random.uniform(0.3, 1.0) if random.random() < 0.4 else 0
            }
            
            # Assign device location
            zone_x = random.uniform(0, 1000)
            zone_y = random.uniform(0, 1000)
            
            # Get device type characteristics
            entropy_range = device_types[device_type]["entropy_range"]
            transmission_interval_range = device_types[device_type]["transmission_interval_range"]
            
            devices.append({
                "device_id": f"dev_{device_id:05d}",
                "device_type": device_type,
                "base_entropy": random.uniform(*entropy_range),
                "transmission_interval": random.uniform(*transmission_interval_range),
                "temporal_weights": temporal_weights,
                "zone_x": zone_x,
                "zone_y": zone_y,
                "last_transmission_time": 0,
                "assigned_fog": None  # Will be assigned later
            })
            device_id += 1
            
    return devices

def initialize_fog_nodes(num_fog_nodes, devices, total_bandwidth, total_energy):
    """
    Initialize fog nodes and assign devices to them.
    
    Args:
        num_fog_nodes (int): Number of fog nodes to create
        devices (list): List of device objects
        total_bandwidth (float): Total available bandwidth
        total_energy (float): Total available energy
        
    Returns:
        list: List of fog node objects
    """
    fog_nodes = []
    
    # Create fog nodes
    for i in range(num_fog_nodes):
        # Position the fog node
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        coverage_radius = random.uniform(100, 300)
        
        fog_nodes.append({
            "fog_id": f"fog_{i:03d}",
            "x": x,
            "y": y,
            "coverage_radius": coverage_radius,
            "assigned_devices": [],
            "current_bandwidth": total_bandwidth / num_fog_nodes,
            "current_energy": total_energy / num_fog_nodes,
            "max_bandwidth": total_bandwidth / num_fog_nodes,
            "max_energy": total_energy / num_fog_nodes
        })
    
    # Assign devices to closest fog nodes with coverage
    for device in devices:
        device_x = device["zone_x"]
        device_y = device["zone_y"]
        
        # Find closest fog node with coverage
        min_distance = float('inf')
        assigned_fog = None
        
        for fog in fog_nodes:
            distance = math.sqrt((device_x - fog["x"])**2 + (device_y - fog["y"])**2)
            if distance <= fog["coverage_radius"] and distance < min_distance:
                min_distance = distance
                assigned_fog = fog
        
        # If no fog node with coverage, assign to closest one
        if not assigned_fog:
            min_distance = float('inf')
            for fog in fog_nodes:
                distance = math.sqrt((device_x - fog["x"])**2 + (device_y - fog["y"])**2)
                if distance < min_distance:
                    min_distance = distance
                    assigned_fog = fog
        
        # Update device and fog node
        device["assigned_fog"] = assigned_fog["fog_id"]
        assigned_fog["assigned_devices"].append(device["device_id"])
    
    return fog_nodes

def generate_packet_content(entropy, packet_size, alphabet_size):
    """
    Generate synthetic packet content with given entropy level.
    
    Args:
        entropy (float): Entropy level between 0 and 1
        packet_size (int): Size of packet in bytes
        alphabet_size (int): Size of the symbol alphabet
        
    Returns:
        tuple: (content_sample, raw_content)
    """
    if entropy <= 0 or entropy >= 1:
        entropy = 0.5  # Default to middle entropy if out of bounds
        
    # Map entropy to Zipf parameter (higher entropy = higher parameter = more uniform)
    zipf_param = 1.0 + 3.0 * (1.0 - entropy)  # Ranges from 1.0 to 4.0
    
    # Generate probabilities following Zipf distribution
    probs = zipf.pmf(range(1, alphabet_size + 1), zipf_param)
    probs = probs / np.sum(probs)
    
    # Generate content based on probability distribution
    content = np.random.choice(alphabet_size, size=packet_size, p=probs)
    
    # Convert to hex string for storage efficiency
    content_sample = ''.join([format(b, '02x') for b in content[:32]])
    
    return content_sample, content.tolist()

def calculate_distance(device, fog_node):
    """Calculate Euclidean distance between device and fog node."""
    return math.sqrt((device["zone_x"] - fog_node["x"])**2 + 
                    (device["zone_y"] - fog_node["y"])**2)
