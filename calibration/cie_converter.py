"""
CIE Color Space Converter

Provides accurate color space conversions following CIE standards
for professional color calibration and analysis.
"""

import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class CIEColorConverter:
    """
    Professional-grade color space converter implementing CIE standards.
    """
    
    def __init__(self, illuminant: str = 'D65'):
        """
        Initialize converter with specified illuminant.
        
        Args:
            illuminant: Reference illuminant ('D65', 'D50', 'A', 'C', 'E')
        """
        self.illuminant = illuminant
        self.white_point = self._get_white_point(illuminant)
        
        # sRGB to XYZ transformation matrix (D65)
        self.srgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # XYZ to sRGB transformation matrix (D65)
        self.xyz_to_srgb_matrix = np.linalg.inv(self.srgb_to_xyz_matrix)
    
    def _get_white_point(self, illuminant: str) -> np.ndarray:
        """Get white point for specified illuminant."""
        white_points = {
            'D65': np.array([95.047, 100.000, 108.883]),
            'D50': np.array([96.422, 100.000, 82.521]),
            'A': np.array([109.850, 100.000, 35.585]),
            'C': np.array([98.074, 100.000, 118.232]),
            'E': np.array([100.000, 100.000, 100.000])
        }
        return white_points.get(illuminant, white_points['D65'])
    
    def srgb_to_linear_rgb(self, srgb: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert sRGB to linear RGB using gamma correction.
        
        Args:
            srgb: sRGB values (0-255 or 0-1)
            
        Returns:
            Linear RGB values (0-1)
        """
        srgb = np.asarray(srgb, dtype=np.float64)
        
        # Normalize to 0-1 if values are in 0-255 range
        if np.max(srgb) > 1.0:
            srgb = srgb / 255.0
        
        # Apply inverse gamma correction
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        
        return linear
    
    def linear_rgb_to_srgb(self, linear_rgb: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert linear RGB to sRGB using gamma correction.
        
        Args:
            linear_rgb: Linear RGB values (0-1)
            
        Returns:
            sRGB values (0-1)
        """
        linear_rgb = np.asarray(linear_rgb, dtype=np.float64)
        
        # Apply gamma correction
        srgb = np.where(
            linear_rgb <= 0.0031308,
            linear_rgb * 12.92,
            1.055 * np.power(linear_rgb, 1.0 / 2.4) - 0.055
        )
        
        return np.clip(srgb, 0, 1)
    
    def srgb_to_xyz(self, srgb: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert sRGB to CIE XYZ color space.
        
        Args:
            srgb: sRGB values (0-255 or 0-1)
            
        Returns:
            XYZ values
        """
        # Convert to linear RGB
        linear_rgb = self.srgb_to_linear_rgb(srgb)
        
        # Apply transformation matrix
        if linear_rgb.ndim == 1:
            xyz = np.dot(self.srgb_to_xyz_matrix, linear_rgb)
        else:
            xyz = np.dot(linear_rgb, self.srgb_to_xyz_matrix.T)
        
        return xyz * 100  # Scale to 0-100 range
    
    def xyz_to_srgb(self, xyz: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert CIE XYZ to sRGB color space.
        
        Args:
            xyz: XYZ values
            
        Returns:
            sRGB values (0-1)
        """
        xyz = np.asarray(xyz, dtype=np.float64) / 100  # Normalize from 0-100 to 0-1
        
        # Apply transformation matrix
        if xyz.ndim == 1:
            linear_rgb = np.dot(self.xyz_to_srgb_matrix, xyz)
        else:
            linear_rgb = np.dot(xyz, self.xyz_to_srgb_matrix.T)
        
        # Convert to sRGB
        srgb = self.linear_rgb_to_srgb(linear_rgb)
        
        return srgb
    
    def xyz_to_lab(self, xyz: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert CIE XYZ to CIE LAB color space.
        
        Args:
            xyz: XYZ values
            
        Returns:
            LAB values
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        
        # Normalize by white point
        xyz_normalized = xyz / self.white_point
        
        # Apply LAB transformation function
        def f(t):
            delta = 6.0 / 29.0
            return np.where(
                t > delta**3,
                np.power(t, 1.0/3.0),
                t / (3 * delta**2) + 4.0/29.0
            )
        
        fx = f(xyz_normalized[..., 0])
        fy = f(xyz_normalized[..., 1])
        fz = f(xyz_normalized[..., 2])
        
        # Calculate LAB values
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        if xyz.ndim == 1:
            return np.array([L, a, b])
        else:
            return np.stack([L, a, b], axis=-1)
    
    def lab_to_xyz(self, lab: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert CIE LAB to CIE XYZ color space.
        
        Args:
            lab: LAB values
            
        Returns:
            XYZ values
        """
        lab = np.asarray(lab, dtype=np.float64)
        
        if lab.ndim == 1:
            L, a, b = lab[0], lab[1], lab[2]
        else:
            L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        
        # Calculate intermediate values
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        # Apply inverse LAB transformation function
        def f_inv(t):
            delta = 6.0 / 29.0
            return np.where(
                t > delta,
                np.power(t, 3),
                3 * delta**2 * (t - 4.0/29.0)
            )
        
        # Calculate XYZ values
        X = self.white_point[0] * f_inv(fx)
        Y = self.white_point[1] * f_inv(fy)
        Z = self.white_point[2] * f_inv(fz)
        
        if lab.ndim == 1:
            return np.array([X, Y, Z])
        else:
            return np.stack([X, Y, Z], axis=-1)
    
    def srgb_to_lab(self, srgb: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert sRGB directly to CIE LAB color space.
        
        Args:
            srgb: sRGB values (0-255 or 0-1)
            
        Returns:
            LAB values
        """
        xyz = self.srgb_to_xyz(srgb)
        return self.xyz_to_lab(xyz)
    
    def lab_to_srgb(self, lab: Union[np.ndarray, list]) -> np.ndarray:
        """
        Convert CIE LAB directly to sRGB color space.
        
        Args:
            lab: LAB values
            
        Returns:
            sRGB values (0-1)
        """
        xyz = self.lab_to_xyz(lab)
        return self.xyz_to_srgb(xyz)
    
    def delta_e_76(self, lab1: Union[np.ndarray, list], lab2: Union[np.ndarray, list]) -> float:
        """
        Calculate CIE Delta E 1976 color difference.
        
        Args:
            lab1: First LAB color
            lab2: Second LAB color
            
        Returns:
            Delta E value
        """
        lab1 = np.asarray(lab1)
        lab2 = np.asarray(lab2)
        
        diff = lab1 - lab2
        return np.sqrt(np.sum(diff**2))
    
    def delta_e_94(self, lab1: Union[np.ndarray, list], lab2: Union[np.ndarray, list],
                   kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
        """
        Calculate CIE Delta E 1994 color difference.
        
        Args:
            lab1: First LAB color
            lab2: Second LAB color
            kL, kC, kH: Weighting factors
            
        Returns:
            Delta E 94 value
        """
        lab1 = np.asarray(lab1)
        lab2 = np.asarray(lab2)
        
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
        
        # Calculate differences
        dL = L1 - L2
        da = a1 - a2
        db = b1 - b2
        
        # Calculate chroma values
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        dC = C1 - C2
        
        # Calculate hue difference
        dH_squared = da**2 + db**2 - dC**2
        dH = np.sqrt(max(0, dH_squared))
        
        # Calculate weighting functions
        SL = 1.0
        SC = 1 + 0.045 * C1
        SH = 1 + 0.015 * C1
        
        # Calculate Delta E 94
        delta_e = np.sqrt(
            (dL / (kL * SL))**2 +
            (dC / (kC * SC))**2 +
            (dH / (kH * SH))**2
        )
        
        return delta_e
    
    def delta_e_2000(self, lab1: Union[np.ndarray, list], lab2: Union[np.ndarray, list],
                     kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
        """
        Calculate CIE Delta E 2000 color difference (most accurate).
        
        Args:
            lab1: First LAB color
            lab2: Second LAB color
            kL, kC, kH: Weighting factors
            
        Returns:
            Delta E 2000 value
        """
        lab1 = np.asarray(lab1)
        lab2 = np.asarray(lab2)
        
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]
        
        # Calculate average values
        L_avg = (L1 + L2) / 2
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_avg = (C1 + C2) / 2
        
        # Calculate G factor
        G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
        
        # Calculate modified a* values
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        # Calculate modified chroma and hue values
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        C_prime_avg = (C1_prime + C2_prime) / 2
        
        h1_prime = np.arctan2(b1, a1_prime) * 180 / np.pi
        h2_prime = np.arctan2(b2, a2_prime) * 180 / np.pi
        
        if h1_prime < 0:
            h1_prime += 360
        if h2_prime < 0:
            h2_prime += 360
        
        # Calculate differences
        dL_prime = L2 - L1
        dC_prime = C2_prime - C1_prime
        
        dh_prime = h2_prime - h1_prime
        if abs(dh_prime) > 180:
            if h2_prime > h1_prime:
                dh_prime -= 360
            else:
                dh_prime += 360
        
        dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2))
        
        # Calculate weighting functions
        T = (1 - 0.17 * np.cos(np.radians((C_prime_avg + 30) / 2 - 6)) +
             0.24 * np.cos(np.radians(2 * (C_prime_avg + 30) / 2)) +
             0.32 * np.cos(np.radians(3 * (C_prime_avg + 30) / 2 + 6)) -
             0.20 * np.cos(np.radians(4 * (C_prime_avg + 30) / 2 - 63)))
        
        SL = 1 + (0.015 * (L_avg - 50)**2) / np.sqrt(20 + (L_avg - 50)**2)
        SC = 1 + 0.045 * C_prime_avg
        SH = 1 + 0.015 * C_prime_avg * T
        
        RT = -2 * np.sqrt(C_prime_avg**7 / (C_prime_avg**7 + 25**7)) * \
             np.sin(np.radians(60 * np.exp(-((C_prime_avg - 275) / 25)**2)))
        
        # Calculate Delta E 2000
        delta_e = np.sqrt(
            (dL_prime / (kL * SL))**2 +
            (dC_prime / (kC * SC))**2 +
            (dH_prime / (kH * SH))**2 +
            RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
        )
        
        return delta_e
    
    def chromatic_adaptation(self, xyz: Union[np.ndarray, list], 
                           source_illuminant: str, target_illuminant: str) -> np.ndarray:
        """
        Perform chromatic adaptation using Bradford transform.
        
        Args:
            xyz: XYZ values under source illuminant
            source_illuminant: Source illuminant name
            target_illuminant: Target illuminant name
            
        Returns:
            XYZ values adapted to target illuminant
        """
        if source_illuminant == target_illuminant:
            return np.asarray(xyz)
        
        # Bradford transformation matrix
        bradford_matrix = np.array([
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296]
        ])
        
        bradford_inv = np.linalg.inv(bradford_matrix)
        
        # Get white points
        source_wp = self._get_white_point(source_illuminant)
        target_wp = self._get_white_point(target_illuminant)
        
        # Transform white points to Bradford space
        source_rgb = np.dot(bradford_matrix, source_wp)
        target_rgb = np.dot(bradford_matrix, target_wp)
        
        # Calculate adaptation matrix
        adaptation_matrix = np.diag(target_rgb / source_rgb)
        
        # Complete transformation matrix
        transform_matrix = np.dot(bradford_inv, np.dot(adaptation_matrix, bradford_matrix))
        
        # Apply transformation
        xyz = np.asarray(xyz)
        if xyz.ndim == 1:
            adapted_xyz = np.dot(transform_matrix, xyz)
        else:
            adapted_xyz = np.dot(xyz, transform_matrix.T)
        
        return adapted_xyz
