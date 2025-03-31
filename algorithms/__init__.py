# algorithms/__init__.py
"""
Network coding algorithms for IoT data transmission optimization.

This package provides implementations of algorithms for optimizing data
transmission in Fog-Cloud IoT architectures, with a focus on entropy-driven
network coding approaches.
"""

from .base import NetworkCodingAlgorithm
from .default import DefaultAlgorithm

__all__ = ['NetworkCodingAlgorithm', 'DefaultAlgorithm']
