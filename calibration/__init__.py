"""
Color Calibration Module for Foundation Color Matcher

This module provides professional-grade color calibration using X-Rite ColorChecker
charts for accurate color analysis and foundation matching.
"""

from .colorchecker_detector import ColorCheckerDetector
from .color_calibrator import ColorCalibrator
from .cie_converter import CIEColorConverter
from .reference_data import COLORCHECKER_REFERENCE

__all__ = [
    'ColorCheckerDetector',
    'ColorCalibrator', 
    'CIEColorConverter',
    'COLORCHECKER_REFERENCE'
]
