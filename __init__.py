# __init__.py
"""
Enhanced Entropy-Driven Network Coding (E-EDNC) Framework

This package provides a simulation framework and implementation of the E-EDNC
algorithm for optimizing data transmission in Fog-Cloud IoT architectures.

The E-EDNC algorithm integrates real-time entropy estimation, adaptive coding
strategies, and hybrid evolutionary-reinforcement learning to optimize bandwidth
utilization, reduce latency, and decrease energy consumption in IoT networks.
"""

from .framework import IoTSimulationFramework
from .algorithms.ednc import EDNCAlgorithm
from .algorithms.default import DefaultAlgorithm
from .utils.data_generator import generate_iot_dataset
from .utils.visualization import (
    plot_performance_metrics,
    plot_entropy_distribution,
    plot_network_topology,
    plot_packet_flow
)

__version__ = '0.1.0'
__author__ = 'IoT Research Team'

__all__ = [
    'IoTSimulationFramework',
    'EDNCAlgorithm',
    'DefaultAlgorithm',
    'generate_iot_dataset',
    'plot_performance_metrics',
    'plot_entropy_distribution',
    'plot_network_topology',
    'plot_packet_flow'
]
