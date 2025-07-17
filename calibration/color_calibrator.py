"""
Color Calibrator

Main calibration engine that uses ColorChecker detection and CIE color conversion
to provide professional-grade color calibration for foundation matching.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import os
from datetime import datetime

from .colorchecker_detector import ColorCheckerDetector
from .cie_converter import CIEColorConverter
from .reference_data import get_reference_values, get_neutral_patches, get_skin_tone_patches

logger = logging.getLogger(__name__)

class ColorCalibrator:
    """
    Professional color calibration system using X-Rite ColorChecker.
    """
    
    def __init__(self, illuminant: str = 'D65'):
        """
        Initialize calibrator with specified illuminant.
        
        Args:
            illuminant: Reference illuminant for calibration
        """
        self.illuminant = illuminant
        self.detector = ColorCheckerDetector()
        self.converter = CIEColorConverter(illuminant)
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_matrix = None
        self.white_balance_factors = None
        self.gamma_correction = None
        self.calibration_quality = None
        self.calibration_metadata = {}
        
    def calibrate_from_image(self, image: np.ndarray, 
                           manual_corners: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        Perform full calibration from ColorChecker image.
        
        Args:
            image: Image containing ColorChecker chart
            manual_corners: Optional manual corner selection
            
        Returns:
            Calibration results and quality metrics
        """
        try:
            # Detect ColorChecker chart
            if manual_corners:
                chart_data = self.detector.manual_selection(image, manual_corners)
            else:
                chart_data = self.detector.detect_chart(image, debug=True)
            
            if chart_data is None:
                return {
                    'success': False,
                    'error': 'Could not detect ColorChecker chart in image',
                    'suggestions': [
                        'Ensure the ColorChecker chart is clearly visible',
                        'Check lighting conditions',
                        'Try manual corner selection',
                        'Ensure chart is not too small or too large in frame'
                    ]
                }
            
            # Validate detection
            validation = self.detector.validate_detection(chart_data['patches'])
            
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"ColorChecker validation failed: {validation.get('error', 'Unknown error')}",
                    'quality_score': validation['quality_score'],
                    'chart_data': chart_data
                }
            
            # Extract measured colors
            measured_colors = self._extract_measured_colors(chart_data['patches'])
            
            # Get reference colors
            reference_colors = get_reference_values('srgb', self.illuminant)
            
            # Calculate calibration matrices
            calibration_result = self._calculate_calibration_matrices(
                measured_colors, reference_colors
            )
            
            # Calculate white balance
            white_balance_result = self._calculate_white_balance(chart_data['patches'])
            
            # Calculate gamma correction
            gamma_result = self._calculate_gamma_correction(chart_data['patches'])
            
            # Store calibration data
            self.calibration_matrix = calibration_result['matrix']
            self.white_balance_factors = white_balance_result['factors']
            self.gamma_correction = gamma_result['gamma']
            self.calibration_quality = validation['quality_score']
            self.is_calibrated = True
            
            # Store metadata
            self.calibration_metadata = {
                'timestamp': datetime.now().isoformat(),
                'illuminant': self.illuminant,
                'chart_area': chart_data.get('chart_area', 0),
                'aspect_ratio': chart_data.get('aspect_ratio', 0),
                'manual_selection': chart_data.get('manual_selection', False),
                'validation': validation
            }
            
            return {
                'success': True,
                'quality_score': validation['quality_score'],
                'calibration_matrix': self.calibration_matrix.tolist(),
                'white_balance_factors': self.white_balance_factors.tolist(),
                'gamma_correction': self.gamma_correction,
                'color_accuracy': calibration_result['accuracy'],
                'metadata': self.calibration_metadata,
                'chart_data': chart_data
            }
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {
                'success': False,
                'error': f'Calibration error: {str(e)}'
            }
    
    def _extract_measured_colors(self, patches: List[Dict]) -> np.ndarray:
        """Extract RGB colors from detected patches."""
        colors = []
        for patch in sorted(patches, key=lambda x: x['index']):
            colors.append(patch['rgb'])
        return np.array(colors)
    
    def _calculate_calibration_matrices(self, measured: np.ndarray, 
                                      reference: np.ndarray) -> Dict:
        """
        Calculate color calibration matrix using least squares optimization.
        
        Args:
            measured: Measured RGB values from image
            reference: Reference RGB values
            
        Returns:
            Calibration matrix and accuracy metrics
        """
        try:
            # Normalize values to 0-1 range
            measured_norm = measured / 255.0 if np.max(measured) > 1 else measured
            reference_norm = reference / 255.0 if np.max(reference) > 1 else reference
            
            # Add bias term for affine transformation
            measured_augmented = np.column_stack([measured_norm, np.ones(len(measured_norm))])
            
            # Solve for transformation matrix using least squares
            # measured * matrix = reference
            matrix, residuals, rank, s = np.linalg.lstsq(
                measured_augmented, reference_norm, rcond=None
            )
            
            # Calculate accuracy metrics
            predicted = np.dot(measured_augmented, matrix)
            errors = np.linalg.norm(predicted - reference_norm, axis=1)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            
            # Calculate R-squared
            ss_res = np.sum((reference_norm - predicted) ** 2)
            ss_tot = np.sum((reference_norm - np.mean(reference_norm, axis=0)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'matrix': matrix,
                'accuracy': {
                    'mean_error': float(mean_error),
                    'max_error': float(max_error),
                    'r_squared': float(r_squared),
                    'residuals': float(np.sum(residuals)) if len(residuals) > 0 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating calibration matrix: {e}")
            # Fallback to identity matrix
            return {
                'matrix': np.eye(4),
                'accuracy': {
                    'mean_error': float('inf'),
                    'max_error': float('inf'),
                    'r_squared': 0.0,
                    'residuals': float('inf')
                }
            }
    
    def _calculate_white_balance(self, patches: List[Dict]) -> Dict:
        """Calculate white balance correction factors."""
        try:
            # Get neutral patches (white to black series)
            neutral_indices = get_neutral_patches()
            
            # Extract neutral patch colors
            neutral_colors = []
            for patch in patches:
                if patch['index'] in neutral_indices:
                    neutral_colors.append(patch['rgb'])
            
            if not neutral_colors:
                return {'factors': np.array([1.0, 1.0, 1.0])}
            
            neutral_colors = np.array(neutral_colors)
            
            # Calculate average of brightest neutral patches (white and light gray)
            bright_neutrals = neutral_colors[:2]  # White and Neutral 8
            avg_neutral = np.mean(bright_neutrals, axis=0)
            
            # Calculate white balance factors
            # Target is equal RGB values (gray)
            target_gray = np.mean(avg_neutral)
            factors = target_gray / avg_neutral
            
            # Normalize factors
            factors = factors / np.max(factors)
            
            return {'factors': factors}
            
        except Exception as e:
            logger.error(f"Error calculating white balance: {e}")
            return {'factors': np.array([1.0, 1.0, 1.0])}
    
    def _calculate_gamma_correction(self, patches: List[Dict]) -> Dict:
        """Calculate gamma correction value."""
        try:
            # Get neutral patches for gamma calculation
            neutral_indices = get_neutral_patches()
            
            measured_values = []
            reference_values = []
            
            reference_rgb = get_reference_values('srgb', self.illuminant)
            
            for patch in patches:
                if patch['index'] in neutral_indices:
                    measured_values.append(np.mean(patch['rgb']))
                    reference_values.append(np.mean(reference_rgb[patch['index']]))
            
            if len(measured_values) < 3:
                return {'gamma': 2.2}  # Default gamma
            
            measured_values = np.array(measured_values) / 255.0
            reference_values = np.array(reference_values) / 255.0
            
            # Fit gamma curve: measured = reference^gamma
            # Take log: log(measured) = gamma * log(reference)
            log_measured = np.log(measured_values + 1e-10)
            log_reference = np.log(reference_values + 1e-10)
            
            # Linear regression to find gamma
            gamma = np.polyfit(log_reference, log_measured, 1)[0]
            
            # Clamp gamma to reasonable range
            gamma = np.clip(gamma, 1.0, 3.0)
            
            return {'gamma': float(gamma)}
            
        except Exception as e:
            logger.error(f"Error calculating gamma correction: {e}")
            return {'gamma': 2.2}
    
    def apply_calibration(self, rgb_values: Union[np.ndarray, list]) -> np.ndarray:
        """
        Apply calibration to RGB values.
        
        Args:
            rgb_values: Input RGB values (0-255 or 0-1)
            
        Returns:
            Calibrated RGB values
        """
        if not self.is_calibrated:
            logger.warning("No calibration applied - calibrator not calibrated")
            return np.asarray(rgb_values)
        
        try:
            rgb = np.asarray(rgb_values, dtype=np.float64)
            
            # Normalize to 0-1 if needed
            if np.max(rgb) > 1.0:
                rgb = rgb / 255.0
            
            # Apply white balance
            if self.white_balance_factors is not None:
                rgb = rgb * self.white_balance_factors
            
            # Apply gamma correction
            if self.gamma_correction is not None:
                rgb = np.power(rgb, 1.0 / self.gamma_correction)
            
            # Apply color calibration matrix
            if self.calibration_matrix is not None:
                # Add bias term
                if rgb.ndim == 1:
                    rgb_augmented = np.append(rgb, 1.0)
                    calibrated = np.dot(rgb_augmented, self.calibration_matrix)
                else:
                    ones = np.ones((rgb.shape[0], 1))
                    rgb_augmented = np.column_stack([rgb, ones])
                    calibrated = np.dot(rgb_augmented, self.calibration_matrix)
            else:
                calibrated = rgb
            
            # Clip to valid range
            calibrated = np.clip(calibrated, 0, 1)
            
            return calibrated
            
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return np.asarray(rgb_values)
    
    def rgb_to_lab_calibrated(self, rgb_values: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert RGB to LAB using calibrated color conversion.
        
        Args:
            rgb_values: Input RGB values
            
        Returns:
            Calibrated LAB values
        """
        # Apply calibration first
        calibrated_rgb = self.apply_calibration(rgb_values)
        
        # Convert to LAB using accurate CIE conversion
        lab_values = self.converter.srgb_to_lab(calibrated_rgb)
        
        return lab_values
    
    def calculate_calibrated_delta_e(self, rgb1: Union[np.ndarray, list], 
                                   rgb2: Union[np.ndarray, list],
                                   method: str = '2000') -> float:
        """
        Calculate Delta E using calibrated color conversion.
        
        Args:
            rgb1: First RGB color
            rgb2: Second RGB color
            method: Delta E method ('76', '94', '2000')
            
        Returns:
            Calibrated Delta E value
        """
        lab1 = self.rgb_to_lab_calibrated(rgb1)
        lab2 = self.rgb_to_lab_calibrated(rgb2)
        
        if method == '76':
            return self.converter.delta_e_76(lab1, lab2)
        elif method == '94':
            return self.converter.delta_e_94(lab1, lab2)
        elif method == '2000':
            return self.converter.delta_e_2000(lab1, lab2)
        else:
            raise ValueError(f"Unknown Delta E method: {method}")
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration data to file.
        
        Args:
            filepath: Path to save calibration file
            
        Returns:
            Success status
        """
        if not self.is_calibrated:
            logger.error("Cannot save - no calibration data available")
            return False
        
        try:
            calibration_data = {
                'illuminant': self.illuminant,
                'calibration_matrix': self.calibration_matrix.tolist(),
                'white_balance_factors': self.white_balance_factors.tolist(),
                'gamma_correction': self.gamma_correction,
                'quality_score': self.calibration_quality,
                'metadata': self.calibration_metadata
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            logger.info(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration data from file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                calibration_data = json.load(f)
            
            self.illuminant = calibration_data['illuminant']
            self.calibration_matrix = np.array(calibration_data['calibration_matrix'])
            self.white_balance_factors = np.array(calibration_data['white_balance_factors'])
            self.gamma_correction = calibration_data['gamma_correction']
            self.calibration_quality = calibration_data['quality_score']
            self.calibration_metadata = calibration_data['metadata']
            self.is_calibrated = True
            
            # Update converter illuminant
            self.converter = CIEColorConverter(self.illuminant)
            
            logger.info(f"Calibration loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False
    
    def get_calibration_info(self) -> Dict:
        """Get current calibration information."""
        if not self.is_calibrated:
            return {'calibrated': False}
        
        return {
            'calibrated': True,
            'illuminant': self.illuminant,
            'quality_score': self.calibration_quality,
            'metadata': self.calibration_metadata
        }
    
    def reset_calibration(self):
        """Reset calibration to uncalibrated state."""
        self.is_calibrated = False
        self.calibration_matrix = None
        self.white_balance_factors = None
        self.gamma_correction = None
        self.calibration_quality = None
        self.calibration_metadata = {}
        
        logger.info("Calibration reset")
