# framework/simulation.py
"""
Main simulation framework for IoT network coding algorithms.

This module provides the core simulation environment for testing
network coding algorithms in Fog-Cloud IoT architectures.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

from algorithms.base import NetworkCodingAlgorithm
from algorithms.default import DefaultAlgorithm
from .entities import initialize_devices, initialize_fog_nodes, generate_packet_content
from .metrics import calculate_network_metrics, generate_feedback

class IoTSimulationFramework:
    """Framework for simulating IoT traffic with pluggable algorithms."""
    
    def __init__(self, algorithm=None, config=None):
        # Load default configuration
        self.config = config or self._load_default_config()
        
        # Set the algorithm
        self.algorithm = algorithm if algorithm else DefaultAlgorithm()
        
        # Initialize algorithm
        self.algorithm.initialize(self.config)
        
        # Initialize data structures
        self.devices = []
        self.fog_nodes = []
        self.feedback_history = []
        self.entropy_history = {}  # device_id -> list of entropy values
        self.performance_metrics = {
            "bandwidth_utilization": [],
            "latency": [],
            "energy_consumption": [],
            "reliability": [],
            "throughput": []
        }
    
    def _load_default_config(self):
        """Load default configuration parameters."""
        return {
            "sliding_window_size": 100,
            "data_alphabet_size": 256,
            "tensor_dimensionality": 3,
            "coding_degrees_range": (2, 10),
            "coding_schemes": ["RLNC", "Fountain", "Simple"],
            "max_latency": 100,  # ms
            "total_bandwidth": 1e9,  # 1 Gbps
            "total_energy": 100,  # J
            "simulation_duration": 10000,  # time steps
            "num_devices": 1000,
            "num_fog_nodes": 50,
            "feedback_frequency": 100,  # time steps
            "device_types": {
                "temperature_sensor": {
                    "entropy_range": (0.1, 0.4),
                    "packet_size_range": (64, 128),
                    "transmission_interval_range": (10, 60)  # seconds
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
            "network_conditions": {
                "normal": {"probability": 0.7, "latency_factor": 1.0, "reliability_factor": 1.0},
                "congested": {"probability": 0.2, "latency_factor": 2.0, "reliability_factor": 0.8},
                "interference": {"probability": 0.1, "latency_factor": 1.5, "reliability_factor": 0.7}
            }
        }
    
    def setup_environment(self):
        """Initialize devices and fog nodes."""
        self.devices = initialize_devices(
            self.config["num_devices"], 
            self.config.get("device_types", {})
        )
        
        self.fog_nodes = initialize_fog_nodes(
            self.config["num_fog_nodes"],
            self.devices,
            self.config["total_bandwidth"],
            self.config["total_energy"]
        )
        
        # Initialize entropy history for each device
        for device in self.devices:
            self.entropy_history[device["device_id"]] = []
    
    def run_simulation(self, output_dir="iot_td_results"):
        """Run the simulation with the configured algorithm."""
        print(f"Starting simulation for {self.config['simulation_duration']} time steps...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment if not already done
        if not self.devices or not self.fog_nodes:
            self.setup_environment()
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=10)
        timestamps = [start_time + timedelta(seconds=i) 
                     for i in range(self.config["simulation_duration"])]
        
        # Storage for packets
        raw_packets = []
        processed_packets = []
        
        # Run simulation
        for time_step in range(self.config["simulation_duration"]):
            if time_step % 1000 == 0:
                print(f"Simulating time step {time_step}/{self.config['simulation_duration']}...")
                
            timestamp = timestamps[time_step]
            time_step_packets = []
            
            # Generate packets for this time step
            for device in self.devices:
                # Check if it's time for this device to transmit
                if time_step - device["last_transmission_time"] >= device["transmission_interval"]:
                    # Calculate entropy based on device characteristics and temporal factors
                    entropy = self._calculate_packet_entropy(device, timestamp)
                    
                    # Add to history
                    self.entropy_history[device["device_id"]].append(entropy)
                    
                    # Determine packet size
                    device_type = device["device_type"]
                    min_size, max_size = self.config["device_types"][device_type]["packet_size_range"]
                    packet_size = random.randint(min_size, max_size)
                    
                    # Generate content
                    content_sample, raw_content = generate_packet_content(
                        entropy, packet_size, self.config["data_alphabet_size"])
                    
                    # Select network condition
                    network_condition = self._select_network_condition()
                    
                    # Create raw packet
                    packet = {
                        "time_step": time_step,
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "device_id": device["device_id"],
                        "device_type": device_type,
                        "fog_node": device["assigned_fog"],
                        "entropy": entropy,
                        "packet_size": packet_size,
                        "content_sample": content_sample,
                        "raw_content": raw_content,
                        "network_condition": network_condition
                    }
                    
                    raw_packets.append(packet)
                    time_step_packets.append(packet)
                    
                    # Update last transmission time
                    device["last_transmission_time"] = time_step
            
            # Process packets using the algorithm
            if time_step_packets:
                # Get available fog nodes for this time step
                available_fogs = self._get_fog_node_resources()
                
                # Group packets by fog node
                fog_packets = self._group_packets_by_fog(time_step_packets)
                
                # Process each fog node's packets
                for fog_id, packets in fog_packets.items():
                    # Apply algorithm to schedule packets
                    constraints = {
                        "bandwidth": available_fogs[fog_id]["bandwidth"],
                        "energy": available_fogs[fog_id]["energy"]
                    }
                    
                    scheduled_packets = self.algorithm.schedule_packets(packets, constraints)
                    
                    # Process each scheduled packet
                    for packet in scheduled_packets:
                        # Estimate actual entropy using algorithm
                        estimated_entropy = self.algorithm.estimate_entropy(
                            packet, self.config["sliding_window_size"])
                        
                        # Predict future entropy
                        future_entropy = self.algorithm.predict_entropy(
                            self.entropy_history[packet["device_id"]]
                        )
                        
                        # Determine coding parameters
                        coding_params = self.algorithm.determine_coding_parameters(
                            estimated_entropy, 
                            packet["network_condition"]
                        )
                        
                        # Calculate metrics
                        metrics = calculate_network_metrics(
                            packet, coding_params, self.config, self.fog_nodes)
                        
                        # Store processed packet
                        processed_packet = {
                            "packet": packet,
                            "estimated_entropy": estimated_entropy,
                            "predicted_entropy": future_entropy,
                            "coding_params": coding_params,
                            "metrics": metrics
                        }
                        
                        processed_packets.append(processed_packet)
                        
                        # Update fog node resources
                        self._update_fog_resources(
                            packet["fog_node"], metrics["bandwidth_usage"], metrics["energy"])
            
            # Generate feedback and adjust algorithm
            if time_step % self.config["feedback_frequency"] == 0 and processed_packets:
                feedback = generate_feedback(time_step, processed_packets)
                if feedback:
                    self.feedback_history.append(feedback)
                    self.algorithm.process_feedback(feedback)
                    
                    # Update performance metrics
                    self._update_performance_metrics(feedback)
                    
                    # Reset fog node resources periodically
                    self._reset_fog_resources()
        
        # Save results
        self._save_results(raw_packets, processed_packets, output_dir)
        
        return {
            "raw_packets": raw_packets,
            "processed_packets": processed_packets,
            "feedback_history": self.feedback_history,
            "performance_metrics": self.performance_metrics
        }
    
    def _calculate_packet_entropy(self, device, timestamp):
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
    
    def _select_network_condition(self):
        """Select a network condition based on configured probabilities."""
        conditions = list(self.config["network_conditions"].keys())
        probabilities = [self.config["network_conditions"][c]["probability"] for c in conditions]
        
        return random.choices(conditions, weights=probabilities, k=1)[0]
    
    def _group_packets_by_fog(self, packets):
        """Group packets by fog node."""
        fog_packets = {}
        for packet in packets:
            fog_id = packet["fog_node"]
            if fog_id not in fog_packets:
                fog_packets[fog_id] = []
            fog_packets[fog_id].append(packet)
        return fog_packets
    
    def _get_fog_node_resources(self):
        """Get available resources for each fog node."""
        resources = {}
        for fog in self.fog_nodes:
            resources[fog["fog_id"]] = {
                "bandwidth": fog["current_bandwidth"],
                "energy": fog["current_energy"]
            }
        return resources
    
    def _update_fog_resources(self, fog_id, bandwidth_usage, energy_usage):
        """Update fog node resources after packet processing."""
        fog = next(f for f in self.fog_nodes if f["fog_id"] == fog_id)
        fog["current_bandwidth"] = max(0, fog["current_bandwidth"] - bandwidth_usage)
        fog["current_energy"] = max(0, fog["current_energy"] - energy_usage)
    
    def _reset_fog_resources(self):
        """Reset fog node resources to initial values."""
        for fog in self.fog_nodes:
            fog["current_bandwidth"] = self.config["total_bandwidth"] / self.config["num_fog_nodes"]
            fog["current_energy"] = self.config["total_energy"] / self.config["num_fog_nodes"]
    
    def _update_performance_metrics(self, feedback):
        """Update performance metrics based on feedback."""
        self.performance_metrics["bandwidth_utilization"].append(feedback["avg_bandwidth"])
        self.performance_metrics["latency"].append(feedback["avg_latency"])
        self.performance_metrics["energy_consumption"].append(feedback["avg_energy"])
        self.performance_metrics["reliability"].append(feedback["avg_reliability"])
        self.performance_metrics["throughput"].append(feedback["avg_throughput"])
    
    def _save_results(self, raw_packets, processed_packets, output_dir):
        """Save simulation results to files."""
        # Save raw packets (sample only to keep file size manageable)
        sample_size = min(10000, len(raw_packets))
        sample_raw = random.sample(raw_packets, sample_size) if len(raw_packets) > sample_size else raw_packets
        
        # Remove raw_content field to reduce file size
        for packet in sample_raw:
            if "raw_content" in packet:
                del packet["raw_content"]
        
        raw_df = pd.DataFrame(sample_raw)
        raw_df.to_csv(os.path.join(output_dir, "raw_packets.csv"), index=False)
        
        # Save processed packets (flattened structure)
        processed_data = []
        sample_size = min(10000, len(processed_packets))
        sample_processed = random.sample(processed_packets, sample_size) if len(processed_packets) > sample_size else processed_packets
        
        for p in sample_processed:
            # Skip raw_content to reduce file size
            if "raw_content" in p["packet"]:
                del p["packet"]["raw_content"]
                
            data = {
                "time_step": p["packet"]["time_step"],
                "timestamp": p["packet"]["timestamp"],
                "device_id": p["packet"]["device_id"],
                "device_type": p["packet"]["device_type"],
                "fog_node": p["packet"]["fog_node"],
                "raw_entropy": p["packet"]["entropy"],
                "estimated_entropy": p["estimated_entropy"],
                "predicted_entropy": p["predicted_entropy"],
                "packet_size": p["packet"]["packet_size"],
                "network_condition": p["packet"]["network_condition"],
                "coding_degree": p["coding_params"]["coding_degree"],
                "coding_scheme": p["coding_params"]["coding_scheme"],
                "latency": p["metrics"]["latency"],
                "reliability": p["metrics"]["reliability"],
                "energy": p["metrics"]["energy"],
                "bandwidth_usage": p["metrics"]["bandwidth_usage"],
                "mutual_information": p["metrics"]["mutual_information"],
                "throughput": p["metrics"]["throughput"]
            }
            processed_data.append(data)
        
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(os.path.join(output_dir, "processed_packets.csv"), index=False)
        
        # Save feedback history
        feedback_df = pd.DataFrame(self.feedback_history)
        feedback_df.to_csv(os.path.join(output_dir, "feedback.csv"), index=False)
        
        # Save performance metrics
        performance_data = []
        for i, time_point in enumerate(range(0, self.config["simulation_duration"], self.config["feedback_frequency"])):
            if i < len(self.performance_metrics["bandwidth_utilization"]):
                data = {
                    "time_step": time_point,
                    "bandwidth_utilization": self.performance_metrics["bandwidth_utilization"][i],
                    "latency": self.performance_metrics["latency"][i],
                    "energy_consumption": self.performance_metrics["energy_consumption"][i],
                    "reliability": self.performance_metrics["reliability"][i],
                    "throughput": self.performance_metrics["throughput"][i]
                }
                performance_data.append(data)
        
        performance_df = pd.DataFrame(performance_data)
        performance_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
        
        # Save metadata
        metadata = {
            "simulation_params": {
                "sliding_window_size": self.config["sliding_window_size"],
                "data_alphabet_size": self.config["data_alphabet_size"],
                "tensor_dimensionality": self.config["tensor_dimensionality"],
                "coding_degrees_range": self.config["coding_degrees_range"],
                "coding_schemes": self.config["coding_schemes"],
                "max_latency": self.config["max_latency"],
                "total_bandwidth": self.config["total_bandwidth"],
                "total_energy": self.config["total_energy"],
                "simulation_duration": self.config["simulation_duration"],
                "num_devices": self.config["num_devices"],
                "num_fog_nodes": self.config["num_fog_nodes"],
                "feedback_frequency": self.config["feedback_frequency"]
            },
            "summary_stats": {
                "total_raw_packets": len(raw_packets),
                "total_processed_packets": len(processed_packets),
                "packets_per_device_type": raw_df["device_type"].value_counts().to_dict(),
                "avg_entropy": raw_df["entropy"].mean(),
                "avg_latency": processed_df["latency"].mean() if not processed_df.empty else None,
                "avg_reliability": processed_df["reliability"].mean() if not processed_df.empty else None,
                "avg_throughput": processed_df["throughput"].mean() if not processed_df.empty else None
            }
        }
        
        with open(os.path.join(output_dir, "simulation_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
