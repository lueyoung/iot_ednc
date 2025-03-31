# utils/data_generator.py
"""
Data generation utilities for IoT simulation.

This module provides functions to generate synthetic IoT datasets
with realistic entropy characteristics for testing algorithms.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import zipf
from collections import defaultdict, Counter

def generate_iot_dataset(config=None, output_dir="iot_td_data"):
    """
    Generate a synthetic IoT Traffic Dataset (IOT-TD) with realistic characteristics.
    
    Args:
        config (dict): Configuration parameters
        output_dir (str): Directory to save generated data
        
    Returns:
        dict: Dictionary containing generated datasets and metadata
    """
    # Load default configuration if not provided
    if config is None:
        config = {
            "num_devices": 1000,
            "num_packets": 100000,
            "time_span_days": 10,
            "data_alphabet_size": 256,
            "device_types": {
                "temperature_sensor": {
                    "entropy_range": (0.1, 0.4),
                    "packet_size_range": (64, 128),
                    "transmission_interval_range": (10, 60)
                },
                "humidity_sensor": {
                    "entropy_range": (0.2, 0.5),
                    "packet_size_range": (64, 128),
                    "transmission_interval_range": (15, 120)
                },
                "motion_detector": {
                    "entropy_range": (0.6, 0.9),
                    "packet_size_range": (128, 256),
                    "transmission_interval_range": (1, 10)
                },
                "camera": {
                    "entropy_range": (0.7, 0.95),
                    "packet_size_range": (1024, 4096),
                    "transmission_interval_range": (5, 30)
                },
                "smart_meter": {
                    "entropy_range": (0.3, 0.6),
                    "packet_size_range": (128, 256),
                    "transmission_interval_range": (300, 900)
                },
                "health_monitor": {
                    "entropy_range": (0.5, 0.8),
                    "packet_size_range": (256, 512),
                    "transmission_interval_range": (60, 300)
                }
            },
            "network_conditions": ["normal", "congested", "interference"],
            "network_condition_probs": [0.7, 0.2, 0.1]
        }
    
    print(f"Generating IoT dataset with {config['num_packets']} packets...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize devices
    devices = _initialize_devices(config["num_devices"], config["device_types"])
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=config["time_span_days"])
    timestamps = [start_time + timedelta(
        seconds=random.randint(0, config["time_span_days"] * 24 * 60 * 60)
    ) for _ in range(config["num_packets"])]
    timestamps.sort()  # Ensure chronological order
    
    # Generate packets
    packets = []
    device_packet_counts = defaultdict(int)
    
    for i in range(config["num_packets"]):
        if i % 10000 == 0:
            print(f"Generated {i} packets...")
        
        # Select a device with less packets first to ensure balance
        eligible_devices = sorted(
            devices, 
            key=lambda d: device_packet_counts[d["device_id"]]
        )
        device = eligible_devices[0]
        device_packet_counts[device["device_id"]] += 1
        
        device_type = device["device_type"]
        timestamp = timestamps[i]
        
        # Calculate entropy based on device and temporal factors
        entropy = _calculate_packet_entropy(device, timestamp)
        
        # Determine packet size
        min_size, max_size = config["device_types"][device_type]["packet_size_range"]
        packet_size = random.randint(min_size, max_size)
        
        # Generate content
        content_sample, raw_content = _generate_packet_content(
            entropy, packet_size, config["data_alphabet_size"])
        
        # Select network condition
        network_condition = random.choices(
            config["network_conditions"], 
            weights=config["network_condition_probs"]
        )[0]
        
        # Add packet to the list
        packet = {
            "timestamp": timestamp,
            "device_id": device["device_id"],
            "device_type": device_type,
            "entropy": entropy,
            "packet_size": packet_size,
            "content_sample": content_sample,
            "network_condition": network_condition,
            "zone_x": device["zone_x"],
            "zone_y": device["zone_y"]
        }
        
        packets.append(packet)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(packets)
    df.to_csv(os.path.join(output_dir, "iot_td_packets.csv"), index=False)
    
    # Save device information
    devices_info = []
    for device in devices:
        devices_info.append({
            "device_id": device["device_id"],
            "device_type": device["device_type"],
            "base_entropy": device["base_entropy"],
            "zone_x": device["zone_x"],
            "zone_y": device["zone_y"]
        })
    
    devices_df = pd.DataFrame(devices_info)
    devices_df.to_csv(os.path.join(output_dir, "iot_td_devices.csv"), index=False)
    
    # Calculate and save entropy distribution
    entropy_bins = np.linspace(0, 1, 21)  # 20 bins
    hist, _ = np.histogram(df["entropy"].values, bins=entropy_bins)
    entropy_dist = pd.DataFrame({
        "entropy_bin_min": entropy_bins[:-1],
        "entropy_bin_max": entropy_bins[1:],
        "packet_count": hist
    })
    entropy_dist.to_csv(os.path.join(output_dir, "iot_td_entropy_dist.csv"), index=False)
    
    # Save metadata
    metadata = {
        "generation_params": config,
        "summary_stats": {
            "total_packets": len(packets),
            "packets_per_device_type": df["device_type"].value_counts().to_dict(),
            "avg_entropy": df["entropy"].mean(),
            "min_entropy": df["entropy"].min(),
            "max_entropy": df["entropy"].max(),
            "std_entropy": df["entropy"].std(),
            "network_condition_counts": df["network_condition"].value_counts().to_dict()
        }
    }
    
    with open(os.path.join(output_dir, "iot_td_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generation complete. Files saved to {output_dir}/")
    print(f"Total packets generated: {len(packets)}")
    
    # Print entropy summary
    entropies = df["entropy"].values
    print("\nEntropy Statistics:")
    print(f"Min: {entropies.min():.4f}")
    print(f"Max: {entropies.max():.4f}")
    print(f"Mean: {entropies.mean():.4f}")
    print(f"Std: {entropies.std():.4f}")
    
    return {
        "packets_df": df,
        "devices_df": devices_df,
        "entropy_dist_df": entropy_dist,
        "metadata": metadata
    }

def _initialize_devices(num_devices, device_types):
    """Initialize IoT devices with realistic characteristics."""
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
            
            devices.append({
                "device_id": f"dev_{device_id:05d}",
                "device_type": device_type,
                "base_entropy": random.uniform(*device_types[device_type]["entropy_range"]),
                "temporal_weights": temporal_weights,
                "zone_x": zone_x,
                "zone_y": zone_y
            })
            device_id += 1
            
    return devices

def _calculate_packet_entropy(device, timestamp):
    """Calculate packet entropy based on device characteristics and temporal factors."""
    # Base entropy from device characteristics
    base_entropy = device["base_entropy"]
    
    # Extract hour and day of week for temporal patterns
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Apply temporal patterns
    daily_effect = 0.2 * np.sin(hour * np.pi / 12) + 0.2
    weekly_effect = 0.1 * np.sin(day_of_week * np.pi / 3.5) + 0.1
    random_spike = 0.3 if random.random() < 0.05 else 0
    
    # Apply weights
    daily_weight = device["temporal_weights"]["daily"]
    weekly_weight = device["temporal_weights"]["weekly"]
    spike_weight = device["temporal_weights"]["random_spike"]
    
    temporal_effect = (
        daily_weight * daily_effect + 
        weekly_weight * weekly_effect + 
        spike_weight * random_spike
    )
    
    # Calculate entropy with temporal effect
    entropy = max(0, min(1, base_entropy + temporal_effect))
    
    return entropy

def _generate_packet_content(entropy, packet_size, alphabet_size):
    """Generate synthetic packet content with given entropy level."""
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

def load_iot_dataset(dataset_dir):
    """
    Load a previously generated IoT dataset.
    
    Args:
        dataset_dir (str): Directory containing dataset files
        
    Returns:
        dict: Dictionary containing loaded datasets
    """
    packets_df = pd.read_csv(os.path.join(dataset_dir, "iot_td_packets.csv"))
    devices_df = pd.read_csv(os.path.join(dataset_dir, "iot_td_devices.csv"))
    entropy_dist_df = pd.read_csv(os.path.join(dataset_dir, "iot_td_entropy_dist.csv"))
    
    with open(os.path.join(dataset_dir, "iot_td_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    return {
        "packets_df": packets_df,
        "devices_df": devices_df,
        "entropy_dist_df": entropy_dist_df,
        "metadata": metadata
    }
