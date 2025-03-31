#!/usr/bin/env python3
"""
Example script for running an E-EDNC simulation.

This script demonstrates how to use the IoT simulation framework
with the E-EDNC algorithm to optimize data transmission in 
Fog-Cloud IoT architectures.
"""

import os
import sys
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use direct imports from the parent directory
from framework import IoTSimulationFramework
from algorithms.ednc import EDNCAlgorithm
from algorithms.default import DefaultAlgorithm
from utils.visualization import (
    plot_performance_metrics,
    plot_entropy_distribution,
    plot_network_topology,
    plot_comparative_results,
    create_metric_summary
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run IoT simulation with E-EDNC algorithm')
    
    parser.add_argument('--algorithm', type=str, default='ednc', choices=['ednc', 'default'],
                        help='Algorithm to use (ednc or default)')
    parser.add_argument('--duration', type=int, default=10000,
                        help='Simulation duration in time steps')
    parser.add_argument('--devices', type=int, default=1000,
                        help='Number of IoT devices')
    parser.add_argument('--fog-nodes', type=int, default=50,
                        help='Number of fog nodes')
    parser.add_argument('--output-dir', type=str, default='simulation_results',
                        help='Directory to save simulation results')
    parser.add_argument('--compare', action='store_true',
                        help='Run both algorithms and compare')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    
    return parser.parse_args()

def run_single_simulation(algorithm_type, config, output_dir):
    """Run a simulation with the specified algorithm."""
    # Create algorithm instance
    if algorithm_type == 'ednc':
        algorithm = EDNCAlgorithm()
        result_dir = os.path.join(output_dir, 'ednc')
    else:
        algorithm = DefaultAlgorithm()
        result_dir = os.path.join(output_dir, 'default')
    
    # Create simulation framework
    framework = IoTSimulationFramework(algorithm, config)
    
    # Set up the environment
    print(f"Setting up environment with {config['num_devices']} devices and {config['num_fog_nodes']} fog nodes...")
    framework.setup_environment()
    
    # Run simulation
    print(f"Running simulation with {algorithm_type} algorithm for {config['simulation_duration']} time steps...")
    start_time = time.time()
    results = framework.run_simulation(output_dir=result_dir)
    elapsed_time = time.time() - start_time
    
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {result_dir}")
    
    return results

def run_comparative_simulation(config, output_dir):
    """Run simulations with both algorithms for comparison."""
    results = {}
    
    # Run with E-EDNC algorithm
    print("Running simulation with E-EDNC algorithm...")
    results['E-EDNC'] = run_single_simulation('ednc', config, output_dir)
    
    # Run with default algorithm
    print("Running simulation with default algorithm...")
    results['Default'] = run_single_simulation('default', config, output_dir)
    
    # Create comparison visualizations
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract performance metrics for comparison
    performance_metrics = {
        'E-EDNC': results['E-EDNC']['performance_metrics'],
        'Default': results['Default']['performance_metrics']
    }
    
    # Create comparative plots
    for metric in ['bandwidth_utilization', 'latency', 'energy_consumption', 
                 'reliability', 'throughput']:
        plot_comparative_results(
            performance_metrics, metric=metric,
            output_file=os.path.join(comparison_dir, f'comparative_{metric}.png')
        )
    
    # Create metric summary
    summary_df = create_metric_summary(
        performance_metrics,
        output_file=os.path.join(comparison_dir, 'metric_summary.csv')
    )
    
    print(f"Comparative analysis results saved to {comparison_dir}")
    print("\nMetric Summary:")
    print(summary_df.to_string())
    
    return results

def create_visualizations(results, output_dir):
    """Create visualizations from simulation results."""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot performance metrics
    plot_performance_metrics(
        results['performance_metrics'],
        output_file=os.path.join(vis_dir, 'performance_metrics.png')
    )
    
    # Create device and fog node dataframes for topology visualization
    devices_data = []
    for device in results.get('devices', []):
        devices_data.append({
            'device_id': device['device_id'],
            'device_type': device['device_type'],
            'zone_x': device['zone_x'],
            'zone_y': device['zone_y']
        })
    
    fog_nodes_data = []
    for fog in results.get('fog_nodes', []):
        fog_nodes_data.append({
            'fog_id': fog['fog_id'],
            'x': fog['x'],
            'y': fog['y'],
            'coverage_radius': fog['coverage_radius']
        })
    
    if devices_data and fog_nodes_data:
        devices_df = pd.DataFrame(devices_data)
        fog_nodes_df = pd.DataFrame(fog_nodes_data)
        
        # Plot network topology
        plot_network_topology(
            devices_df, fog_nodes_df,
            output_file=os.path.join(vis_dir, 'network_topology.png')
        )
    
    # Create processed packets dataframe for visualization
    if 'processed_packets' in results:
        # Load from CSV if saved during simulation
        processed_df = pd.read_csv(os.path.join(output_dir, 'processed_packets.csv'))
        
        # Plot packet flow visualization
        plot_packet_flow(
            processed_df,
            output_file=os.path.join(vis_dir, 'packet_flow.png')
        )
    
    print(f"Visualizations saved to {vis_dir}")

def main():
    """Main function to run the simulation."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure simulation
    config = {
        "simulation_duration": args.duration,
        "num_devices": args.devices,
        "num_fog_nodes": args.fog_nodes,
        "sliding_window_size": 100,
        "data_alphabet_size": 256,
        "tensor_dimensionality": 3,
        "coding_degrees_range": (2, 10),
        "coding_schemes": ["RLNC", "Fountain", "Simple"],
        "max_latency": 100,  # ms
        "total_bandwidth": 1e9,  # 1 Gbps
        "total_energy": 100,  # J
        "feedback_frequency": 100  # time steps
    }
    
    # Run simulation(s)
    if args.compare:
        results = run_comparative_simulation(config, args.output_dir)
    else:
        results = run_single_simulation(args.algorithm, config, args.output_dir)
    
    # Generate visualizations if requested
    if args.visualize:
        if args.compare:
            # Visualizations for both algorithms already created in comparison
            pass
        else:
            create_visualizations(results, os.path.join(args.output_dir, args.algorithm))

    print("Done!")

if __name__ == "__main__":
    main()
