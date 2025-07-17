"""
ColorChecker Detection Module

Automatically detects X-Rite ColorChecker charts in images and extracts
individual color patches for calibration purposes.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class ColorCheckerDetector:
    """
    Detects and extracts ColorChecker charts from images using computer vision.
    """
    
    def __init__(self):
        self.chart_aspect_ratio = 4.0 / 6.0  # ColorChecker is 4 patches wide, 6 tall
        self.min_chart_area = 10000  # Minimum area for chart detection
        self.max_chart_area = 500000  # Maximum area for chart detection
        
    def detect_chart(self, image: np.ndarray, debug: bool = False) -> Optional[Dict]:
        """
        Detect ColorChecker chart in image and return chart information.
        
        Args:
            image: Input image (BGR format)
            debug: Whether to return debug information
            
        Returns:
            Dictionary containing chart corners, transformation matrix, and patches
            or None if no chart detected
        """
        try:
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            chart_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_chart_area < area < self.max_chart_area:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it's roughly rectangular (4 corners)
                    if len(approx) >= 4:
                        # Calculate bounding rectangle
                        rect = cv2.boundingRect(approx)
                        aspect_ratio = rect[2] / rect[3]  # width / height
                        
                        # Check if aspect ratio matches ColorChecker
                        if 0.5 < aspect_ratio < 0.8:  # Allow some tolerance
                            chart_candidates.append({
                                'contour': contour,
                                'approx': approx,
                                'area': area,
                                'rect': rect,
                                'aspect_ratio': aspect_ratio
                            })
            
            if not chart_candidates:
                logger.warning("No ColorChecker chart candidates found")
                return None
            
            # Select best candidate (largest area with good aspect ratio)
            best_candidate = max(chart_candidates, 
                               key=lambda x: x['area'] * (1 - abs(x['aspect_ratio'] - self.chart_aspect_ratio)))
            
            # Extract chart corners
            corners = self._extract_corners(best_candidate['approx'])
            if corners is None:
                logger.warning("Could not extract chart corners")
                return None
            
            # Calculate perspective transformation
            transform_matrix = self._calculate_transform_matrix(corners)
            
            # Extract color patches
            patches = self._extract_patches(image, corners, transform_matrix)
            
            result = {
                'corners': corners,
                'transform_matrix': transform_matrix,
                'patches': patches,
                'chart_area': best_candidate['area'],
                'aspect_ratio': best_candidate['aspect_ratio']
            }
            
            if debug:
                result['debug'] = {
                    'candidates': chart_candidates,
                    'edges': edges,
                    'contours': contours
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting ColorChecker chart: {e}")
            return None
    
    def _extract_corners(self, approx_contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the four corners of the ColorChecker chart from approximated contour.
        
        Args:
            approx_contour: Approximated contour points
            
        Returns:
            Array of 4 corner points in order: top-left, top-right, bottom-right, bottom-left
        """
        if len(approx_contour) < 4:
            return None
        
        # Reshape contour points
        points = approx_contour.reshape(-1, 2).astype(np.float32)
        
        # If we have more than 4 points, find the 4 corner points
        if len(points) > 4:
            # Find convex hull
            hull = cv2.convexHull(points)
            hull_points = hull.reshape(-1, 2).astype(np.float32)
            
            # Approximate to 4 corners
            epsilon = 0.02 * cv2.arcLength(hull, True)
            corners = cv2.approxPolyDP(hull, epsilon, True)
            corners = corners.reshape(-1, 2).astype(np.float32)
            
            if len(corners) != 4:
                # Fallback: use bounding rectangle corners
                rect = cv2.boundingRect(points)
                corners = np.array([
                    [rect[0], rect[1]],  # top-left
                    [rect[0] + rect[2], rect[1]],  # top-right
                    [rect[0] + rect[2], rect[1] + rect[3]],  # bottom-right
                    [rect[0], rect[1] + rect[3]]  # bottom-left
                ], dtype=np.float32)
        else:
            corners = points
        
        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(corners)
        
        return corners
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corner points
        """
        # Calculate center point
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        def angle_from_center(point):
            return np.arctan2(point[1] - center[1], point[0] - center[0])
        
        # Sort corners by angle
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # Find top-left corner (minimum x + y)
        sums = [pt[0] + pt[1] for pt in sorted_corners]
        top_left_idx = np.argmin(sums)
        
        # Reorder starting from top-left, going clockwise
        ordered = []
        for i in range(4):
            ordered.append(sorted_corners[(top_left_idx + i) % 4])
        
        return np.array(ordered, dtype=np.float32)
    
    def _calculate_transform_matrix(self, corners: np.ndarray) -> np.ndarray:
        """
        Calculate perspective transformation matrix to normalize the chart.
        
        Args:
            corners: Chart corner points
            
        Returns:
            3x3 transformation matrix
        """
        # Define target rectangle (normalized coordinates)
        target_width = 400
        target_height = 600  # 4:6 aspect ratio
        
        target_corners = np.array([
            [0, 0],  # top-left
            [target_width, 0],  # top-right
            [target_width, target_height],  # bottom-right
            [0, target_height]  # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transformation
        transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
        
        return transform_matrix
    
    def _extract_patches(self, image: np.ndarray, corners: np.ndarray, 
                        transform_matrix: np.ndarray) -> List[Dict]:
        """
        Extract individual color patches from the detected chart.
        
        Args:
            image: Original image
            corners: Chart corner points
            transform_matrix: Perspective transformation matrix
            
        Returns:
            List of patch dictionaries with RGB values and positions
        """
        # Apply perspective transformation
        target_width = 400
        target_height = 600
        warped = cv2.warpPerspective(image, transform_matrix, (target_width, target_height))
        
        patches = []
        
        # ColorChecker layout: 4 columns, 6 rows
        cols, rows = 4, 6
        patch_width = target_width // cols
        patch_height = target_height // rows
        
        # Extract each patch
        for row in range(rows):
            for col in range(cols):
                # Calculate patch boundaries
                x1 = col * patch_width + patch_width // 4  # Skip border
                y1 = row * patch_height + patch_height // 4
                x2 = (col + 1) * patch_width - patch_width // 4
                y2 = (row + 1) * patch_height - patch_height // 4
                
                # Extract patch region
                patch_region = warped[y1:y2, x1:x2]
                
                if patch_region.size > 0:
                    # Calculate average color (BGR to RGB)
                    avg_color_bgr = np.mean(patch_region.reshape(-1, 3), axis=0)
                    avg_color_rgb = avg_color_bgr[::-1]  # BGR to RGB
                    
                    patch_index = row * cols + col
                    
                    patches.append({
                        'index': patch_index,
                        'row': row,
                        'col': col,
                        'rgb': avg_color_rgb,
                        'bgr': avg_color_bgr,
                        'region': patch_region,
                        'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                    })
        
        return patches
    
    def manual_selection(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> Optional[Dict]:
        """
        Extract patches from manually selected chart corners.
        
        Args:
            image: Input image
            corners: List of 4 corner points [(x, y), ...]
            
        Returns:
            Dictionary with chart information or None if invalid
        """
        if len(corners) != 4:
            logger.error("Manual selection requires exactly 4 corner points")
            return None
        
        try:
            corners_array = np.array(corners, dtype=np.float32)
            corners_array = self._order_corners(corners_array)
            
            transform_matrix = self._calculate_transform_matrix(corners_array)
            patches = self._extract_patches(image, corners_array, transform_matrix)
            
            return {
                'corners': corners_array,
                'transform_matrix': transform_matrix,
                'patches': patches,
                'manual_selection': True
            }
            
        except Exception as e:
            logger.error(f"Error in manual selection: {e}")
            return None
    
    def validate_detection(self, patches: List[Dict], tolerance: float = 30.0) -> Dict:
        """
        Validate detected patches against known ColorChecker values.
        
        Args:
            patches: Extracted color patches
            tolerance: Maximum allowed color difference for validation
            
        Returns:
            Validation results with quality metrics
        """
        from .reference_data import get_reference_values
        
        if len(patches) != 24:
            return {
                'valid': False,
                'error': f'Expected 24 patches, found {len(patches)}',
                'quality_score': 0.0
            }
        
        try:
            # Get reference sRGB values
            reference_rgb = get_reference_values('srgb')
            
            # Calculate color differences
            differences = []
            for i, patch in enumerate(patches):
                if i < len(reference_rgb):
                    patch_rgb = patch['rgb']
                    ref_rgb = reference_rgb[i]
                    
                    # Calculate Euclidean distance in RGB space
                    diff = np.linalg.norm(patch_rgb - ref_rgb)
                    differences.append(diff)
            
            avg_difference = np.mean(differences)
            max_difference = np.max(differences)
            
            # Calculate quality score (0-100)
            quality_score = max(0, 100 - (avg_difference / tolerance) * 100)
            
            return {
                'valid': avg_difference < tolerance,
                'average_difference': avg_difference,
                'max_difference': max_difference,
                'quality_score': quality_score,
                'patch_differences': differences
            }
            
        except Exception as e:
            logger.error(f"Error validating detection: {e}")
            return {
                'valid': False,
                'error': str(e),
                'quality_score': 0.0
            }
