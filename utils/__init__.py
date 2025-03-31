# utils/__init__.py
"""
Utility functions for the IoT simulation framework.

This package provides data generation and visualization utilities
to support the simulation and analysis of IoT network coding algorithms.
"""

from .data_generator import generate_iot_dataset
from .visualization import (
    plot_performance_metrics,
    plot_entropy_distribution,
    plot_network_topology,
    plot_packet_flow
)

__all__ = [
    'generate_iot_dataset',
    'plot_performance_metrics',
    'plot_entropy_distribution',
    'plot_network_topology',
    'plot_packet_flow'
]
