"""
X-Rite ColorChecker Reference Data

Contains the official reference values for X-Rite ColorChecker Classic 24-patch chart
in various color spaces (sRGB, XYZ, LAB) under standard illuminants.
"""

import numpy as np

# X-Rite ColorChecker Classic 24-patch reference values
# Values are under D65 illuminant, 2° observer
COLORCHECKER_REFERENCE = {
    'names': [
        'Dark Skin', 'Light Skin', 'Blue Sky', 'Foliage', 'Blue Flower', 'Bluish Green',
        'Orange', 'Purplish Blue', 'Moderate Red', 'Purple', 'Yellow Green', 'Orange Yellow',
        'Blue', 'Green', 'Red', 'Yellow', 'Magenta', 'Cyan',
        'White', 'Neutral 8', 'Neutral 6.5', 'Neutral 5', 'Neutral 3.5', 'Black'
    ],
    
    # sRGB values (0-255) under D65
    'srgb': np.array([
        [115, 82, 68],    # Dark Skin
        [194, 150, 130],  # Light Skin
        [98, 122, 157],   # Blue Sky
        [87, 108, 67],    # Foliage
        [133, 128, 177],  # Blue Flower
        [103, 189, 170],  # Bluish Green
        [214, 126, 44],   # Orange
        [80, 91, 166],    # Purplish Blue
        [193, 90, 99],    # Moderate Red
        [94, 60, 108],    # Purple
        [157, 188, 64],   # Yellow Green
        [224, 163, 46],   # Orange Yellow
        [56, 61, 150],    # Blue
        [70, 148, 73],    # Green
        [175, 54, 60],    # Red
        [231, 199, 31],   # Yellow
        [187, 86, 149],   # Magenta
        [8, 133, 161],    # Cyan
        [243, 243, 242],  # White
        [200, 200, 200],  # Neutral 8
        [160, 160, 160],  # Neutral 6.5
        [122, 122, 121],  # Neutral 5
        [85, 85, 85],     # Neutral 3.5
        [52, 52, 52]      # Black
    ], dtype=np.float64),
    
    # CIE XYZ values under D65 illuminant
    'xyz': np.array([
        [10.1, 9.0, 5.1],     # Dark Skin
        [35.8, 35.6, 29.1],   # Light Skin
        [19.3, 20.1, 42.1],   # Blue Sky
        [13.3, 15.3, 9.3],    # Foliage
        [24.3, 20.9, 46.9],   # Blue Flower
        [26.9, 36.2, 40.6],   # Bluish Green
        [51.1, 42.0, 13.9],   # Orange
        [11.1, 10.9, 35.8],   # Purplish Blue
        [35.0, 24.3, 18.1],   # Moderate Red
        [12.0, 8.7, 17.9],    # Purple
        [44.3, 49.1, 15.3],   # Yellow Green
        [56.1, 47.4, 13.8],   # Orange Yellow
        [8.1, 7.2, 31.2],     # Blue
        [19.8, 27.4, 17.9],   # Green
        [28.8, 18.0, 9.5],    # Red
        [59.1, 62.6, 17.7],   # Yellow
        [30.0, 21.4, 30.5],   # Magenta
        [20.9, 25.4, 46.9],   # Cyan
        [88.6, 93.2, 104.8],  # White
        [59.1, 62.1, 67.5],   # Neutral 8
        [36.2, 38.1, 40.6],   # Neutral 6.5
        [20.5, 21.5, 22.7],   # Neutral 5
        [9.0, 9.5, 9.9],      # Neutral 3.5
        [3.1, 3.3, 3.3]       # Black
    ], dtype=np.float64),
    
    # CIE LAB values under D65 illuminant
    'lab': np.array([
        [37.99, 13.56, 14.06],   # Dark Skin
        [65.71, 18.13, 17.81],   # Light Skin
        [49.93, -4.88, -21.93],  # Blue Sky
        [43.14, -13.10, 21.61],  # Foliage
        [55.11, 8.84, -25.40],   # Blue Flower
        [70.72, -33.40, -0.20],  # Bluish Green
        [62.66, 36.07, 57.10],   # Orange
        [40.02, 10.41, -45.96],  # Purplish Blue
        [51.12, 48.24, 16.25],   # Moderate Red
        [30.33, 22.98, -21.59],  # Purple
        [72.53, -23.71, 57.26],  # Yellow Green
        [71.94, 19.36, 67.86],   # Orange Yellow
        [28.78, 14.18, -50.30],  # Blue
        [55.26, -38.34, 31.37],  # Green
        [42.10, 53.38, 28.19],   # Red
        [81.73, -4.90, 79.89],   # Yellow
        [51.94, 49.99, -14.57],  # Magenta
        [51.04, -28.63, -28.64], # Cyan
        [96.54, -0.43, 1.19],    # White
        [81.26, -0.64, -0.34],   # Neutral 8
        [66.77, -0.73, -0.50],   # Neutral 6.5
        [50.87, -0.15, -0.27],   # Neutral 5
        [35.66, -0.42, -1.23],   # Neutral 3.5
        [20.46, 0.08, -0.97]     # Black
    ], dtype=np.float64),
    
    # Patch positions in standard ColorChecker layout (4x6 grid)
    # Coordinates are normalized (0-1) for the chart area
    'positions': np.array([
        # Row 1
        [0.125, 0.167], [0.375, 0.167], [0.625, 0.167], [0.875, 0.167],
        # Row 2  
        [0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5],
        # Row 3
        [0.125, 0.833], [0.375, 0.833], [0.625, 0.833], [0.875, 0.833],
        # Row 4
        [0.125, 0.167], [0.375, 0.167], [0.625, 0.167], [0.875, 0.167],
        # Row 5
        [0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5],
        # Row 6
        [0.125, 0.833], [0.375, 0.833], [0.625, 0.833], [0.875, 0.833]
    ], dtype=np.float64)
}

# Standard illuminants (CIE XYZ coordinates)
ILLUMINANTS = {
    'D65': np.array([95.047, 100.000, 108.883]),  # Daylight 6500K
    'D50': np.array([96.422, 100.000, 82.521]),   # Daylight 5000K
    'A': np.array([109.850, 100.000, 35.585]),    # Incandescent
    'C': np.array([98.074, 100.000, 118.232]),    # Average daylight
    'E': np.array([100.000, 100.000, 100.000])    # Equal energy
}

# Color temperature to illuminant mapping
COLOR_TEMPERATURES = {
    2856: 'A',      # Incandescent
    5000: 'D50',    # Horizon daylight
    6500: 'D65',    # Noon daylight
    6774: 'C'       # Average daylight
}

def get_reference_values(color_space='lab', illuminant='D65'):
    """
    Get ColorChecker reference values for specified color space and illuminant.
    
    Args:
        color_space (str): 'srgb', 'xyz', or 'lab'
        illuminant (str): 'D65', 'D50', 'A', 'C', or 'E'
    
    Returns:
        np.ndarray: Reference values for all 24 patches
    """
    if color_space.lower() not in ['srgb', 'xyz', 'lab']:
        raise ValueError("color_space must be 'srgb', 'xyz', or 'lab'")
    
    if illuminant not in ILLUMINANTS:
        raise ValueError(f"illuminant must be one of {list(ILLUMINANTS.keys())}")
    
    # For now, return D65 values (chromatic adaptation can be added later)
    return COLORCHECKER_REFERENCE[color_space.lower()].copy()

def get_patch_name(patch_index):
    """Get the name of a ColorChecker patch by index (0-23)."""
    if 0 <= patch_index < 24:
        return COLORCHECKER_REFERENCE['names'][patch_index]
    else:
        raise ValueError("patch_index must be between 0 and 23")

def get_neutral_patches():
    """Get indices of neutral (gray) patches for white balance."""
    # Patches 18-23 are the neutral series (White to Black)
    return list(range(18, 24))

def get_skin_tone_patches():
    """Get indices of skin tone patches for skin color calibration."""
    # Patches 0 (Dark Skin) and 1 (Light Skin)
    return [0, 1]
