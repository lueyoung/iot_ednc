# utils/visualization.py
"""
Visualization utilities for IoT simulation results.

This module provides functions to create visualizations of simulation
results, including performance metrics, network topology, and entropy
distributions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_performance_metrics(metrics, output_file=None, figsize=(12, 10)):
    """
    Plot performance metrics from simulation results.
    
    Args:
        metrics (dict): Performance metrics dictionary
        output_file (str): File to save the plot (optional)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    time_steps = range(0, len(metrics["bandwidth_utilization"]))
    
    # Plot bandwidth utilization
    axes[0, 0].plot(time_steps, metrics["bandwidth_utilization"], 'b-')
    axes[0, 0].set_title('Bandwidth Utilization')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Bandwidth (bps)')
    axes[0, 0].grid(True)
    
    # Plot latency
    axes[0, 1].plot(time_steps, metrics["latency"], 'r-')
    axes[0, 1].set_title('Latency')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].grid(True)
    
    # Plot energy consumption
    axes[1, 0].plot(time_steps, metrics["energy_consumption"], 'g-')
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Energy (J)')
    axes[1, 0].grid(True)
    
    # Plot reliability
    axes[1, 1].plot(time_steps, metrics["reliability"], 'm-')
    axes[1, 1].set_title('Reliability')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Reliability (0-1)')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True)
    
    # Plot throughput
    axes[2, 0].plot(time_steps, metrics["throughput"], 'c-')
    axes[2, 0].set_title('Throughput')
    axes[2, 0].set_xlabel('Time Step')
    axes[2, 0].set_ylabel('Throughput (Mbps)')
    axes[2, 0].grid(True)
    
    # Plot all metrics normalized
    ax = axes[2, 1]
    metrics_to_plot = ["bandwidth_utilization", "reliability", "throughput"]
    
    for metric in metrics_to_plot:
        # Normalize to [0,1]
        values = metrics[metric]
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        ax.plot(time_steps, normalized, label=metric)
    
    # Invert latency and energy (lower is better)
    for metric in ["latency", "energy_consumption"]:
        values = metrics[metric]
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        ax.plot(time_steps, 1 - normalized, label=f"inv_{metric}")
    
    ax.set_title('Normalized Metrics')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {output_file}")
    
    return fig

def plot_entropy_distribution(entropy_data, output_file=None, figsize=(10, 6)):
    """
    Plot entropy distribution from simulation results.
    
    Args:
        entropy_data (pd.DataFrame): Entropy distribution data
        output_file (str): File to save the plot (optional)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bin centers
    bin_centers = (entropy_data["entropy_bin_min"] + entropy_data["entropy_bin_max"]) / 2
    
    # Create bar chart
    ax.bar(bin_centers, entropy_data["packet_count"], 
           width=(entropy_data["entropy_bin_max"] - entropy_data["entropy_bin_min"]).iloc[0],
           alpha=0.7, color='skyblue', edgecolor='navy')
    
    ax.set_title('Entropy Distribution of IoT Data')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Packet Count')
    ax.grid(True, alpha=0.3)
    
    # Add a line showing the PDF of the distribution
    ax2 = ax.twinx()
    sns.kdeplot(x=bin_centers, weights=entropy_data["packet_count"], 
                color='crimson', linewidth=2, ax=ax2)
    ax2.set_ylabel('Probability Density')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Entropy distribution plot saved to {output_file}")
    
    return fig

def plot_network_topology(devices_df, fog_nodes_df=None, connections=None, 
                         output_file=None, figsize=(12, 10)):
    """
    Plot network topology showing devices and fog nodes.
    
    Args:
        devices_df (pd.DataFrame): Device information
        fog_nodes_df (pd.DataFrame): Fog node information (optional)
        connections (list): List of (device_id, fog_id) connections (optional)
        output_file (str): File to save the plot (optional)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color mapping for device types
    device_types = devices_df["device_type"].unique()
    device_colors = cm.tab10(np.linspace(0, 1, len(device_types)))
    color_map = dict(zip(device_types, device_colors))
    
    # Plot devices
    for device_type in device_types:
        subset = devices_df[devices_df["device_type"] == device_type]
        ax.scatter(subset["zone_x"], subset["zone_y"], 
                  c=[color_map[device_type]], label=device_type,
                  alpha=0.6, s=30)
    
    # Plot fog nodes if available
    if fog_nodes_df is not None:
        ax.scatter(fog_nodes_df["x"], fog_nodes_df["y"], 
                  c='black', marker='s', s=100, label='Fog Node')
        
        # Plot coverage areas
        for _, fog in fog_nodes_df.iterrows():
            circle = plt.Circle((fog["x"], fog["y"]), 
                               fog["coverage_radius"], 
                               fill=False, color='gray', linestyle='--', alpha=0.3)
            ax.add_patch(circle)
    
    # Plot connections if available
    if connections is not None and fog_nodes_df is not None:
        # Convert DataFrames to dictionaries for faster lookup
        device_dict = devices_df.set_index('device_id').to_dict('index')
        fog_dict = fog_nodes_df.set_index('fog_id').to_dict('index')
        
        # Plot a sample of connections to avoid overcrowding
        sample_size = min(500, len(connections))
        sampled_connections = np.random.choice(len(connections), sample_size, replace=False)
        
        for idx in sampled_connections:
            device_id, fog_id = connections[idx]
            if device_id in device_dict and fog_id in fog_dict:
                device = device_dict[device_id]
                fog = fog_dict[fog_id]
                ax.plot([device["zone_x"], fog["x"]], 
                       [device["zone_y"], fog["y"]], 
                       'k-', alpha=0.05)
    
    ax.set_title('IoT Network Topology')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Network topology plot saved to {output_file}")
    
    return fig

def plot_packet_flow(processed_packets, output_file=None, figsize=(12, 8)):
    """
    Visualize packet flow with coding parameters.
    
    Args:
        processed_packets (pd.DataFrame): Processed packet information
        output_file (str): File to save the plot (optional)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Create time series of packet properties
    time_steps = processed_packets["time_step"].values
    entropies = processed_packets["raw_entropy"].values
    coding_degrees = processed_packets["coding_degree"].values
    
    # Plot entropy vs coding degree
    sc = ax1.scatter(time_steps, entropies, c=coding_degrees, 
                   cmap='viridis', alpha=0.6, s=30)
    ax1.set_title('Packet Entropy and Coding Degree Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Entropy')
    ax1.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('Coding Degree')
    
    # Group by coding scheme and visualize
    scheme_counts = processed_packets["coding_scheme"].value_counts()
    ax2.pie(scheme_counts, labels=scheme_counts.index, autopct='%1.1f%%',
           startangle=90, colors=plt.cm.Set3.colors)
    ax2.axis('equal')
    ax2.set_title('Distribution of Coding Schemes')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Packet flow visualization saved to {output_file}")
    
    return fig

def plot_comparative_results(results_dict, metric="latency", output_file=None, figsize=(10, 6)):
    """
    Plot comparative results of multiple algorithms.
    
    Args:
        results_dict (dict): Dictionary of {algorithm_name: metrics}
        metric (str): Metric to compare
        output_file (str): File to save the plot (optional)
        figsize (tuple): Figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each algorithm's results
    for name, metrics in results_dict.items():
        if metric in metrics:
            time_steps = range(len(metrics[metric]))
            ax.plot(time_steps, metrics[metric], label=name)
    
    metric_title = ' '.join(word.capitalize() for word in metric.split('_'))
    ax.set_title(f'Comparative {metric_title}')
    ax.set_xlabel('Time Step')
    
    # Set y-axis label based on metric
    if metric == "latency":
        ax.set_ylabel('Latency (ms)')
    elif metric == "bandwidth_utilization":
        ax.set_ylabel('Bandwidth (bps)')
    elif metric == "energy_consumption":
        ax.set_ylabel('Energy (J)')
    elif metric == "reliability":
        ax.set_ylabel('Reliability (0-1)')
        ax.set_ylim(0, 1)
    elif metric == "throughput":
        ax.set_ylabel('Throughput (Mbps)')
    else:
        ax.set_ylabel(metric_title)
    
    ax.grid(True)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparative results plot saved to {output_file}")
    
    return fig

def create_metric_summary(results_dict, output_file=None):
    """
    Create a summary table of metrics for different algorithms.
    
    Args:
        results_dict (dict): Dictionary of {algorithm_name: metrics}
        output_file (str): File to save the CSV (optional)
        
    Returns:
        pd.DataFrame: Summary table
    """
    summary_data = []
    
    for name, metrics in results_dict.items():
        row = {'Algorithm': name}
        
        # Calculate average for each metric
        for metric in ['bandwidth_utilization', 'latency', 'energy_consumption', 
                     'reliability', 'throughput']:
            if metric in metrics and len(metrics[metric]) > 0:
                row[f'Avg_{metric}'] = np.mean(metrics[metric])
                row[f'Std_{metric}'] = np.std(metrics[metric])
                row[f'Min_{metric}'] = np.min(metrics[metric])
                row[f'Max_{metric}'] = np.max(metrics[metric])
        
        summary_data.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"Metric summary saved to {output_file}")
    
    return summary_df
