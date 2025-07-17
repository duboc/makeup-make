from flask import Flask, render_template, request, jsonify, session, send_file, redirect
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
import os
import uuid
from werkzeug.utils import secure_filename
import json
import logging

# Import calibration modules
from calibration import ColorCalibrator, CIEColorConverter
from mac_foundation_database import convert_mac_database_to_app_format
from natura_foundation_database import convert_natura_database_to_app_format

app = Flask(__name__)

# Production-ready configuration
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Cloud Run specific configurations
# Only require secure cookies in production (HTTPS)
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/calibration_profiles', exist_ok=True)

class FoundationColorPredictor:
    def __init__(self):
        self.skin_detector = SkinDetector()
        self.color_calibrator = ColorCalibrator()
        self.prediction_model = None
        self.foundation_databases = {
            'MAC': None,
            'Natura': None
        }
        self.current_brand = 'MAC'  # Default brand
        self.load_foundation_databases()
    
    def load_foundation_databases(self):
        """Load foundation databases for all supported brands"""
        self.foundation_databases['MAC'] = convert_mac_database_to_app_format()
        self.foundation_databases['Natura'] = convert_natura_database_to_app_format()
    
    def set_brand(self, brand):
        """Set the current brand for foundation matching"""
        if brand in self.foundation_databases:
            self.current_brand = brand
            return True
        return False
    
    def get_current_database(self):
        """Get the current brand's foundation database"""
        return self.foundation_databases.get(self.current_brand, {})
    
    def get_available_brands(self):
        """Get list of available brands"""
        return list(self.foundation_databases.keys())
    
    def predict_foundation_match(self, skin_lab, foundation_lab):
        """Predict the resulting skin color when foundation is applied"""
        if self.prediction_model is None:
            # Use a simple blending model if no trained model exists
            alpha = 0.7  # Foundation coverage factor
            predicted_lab = alpha * np.array(foundation_lab) + (1 - alpha) * np.array(skin_lab)
            return predicted_lab.tolist()
        
        # Use trained model
        input_features = np.array([skin_lab + foundation_lab]).reshape(1, -1)
        return self.prediction_model.predict(input_features)[0]
    
    def find_best_foundation_matches(self, skin_lab, num_matches=5):
        """Find the best foundation matches for given skin color"""
        matches = []
        
        # Detect undertone from skin LAB values
        detected_undertone = self.detect_undertone(skin_lab)
        
        # Get current brand's database
        current_database = self.get_current_database()
        
        for tone_category, foundations in current_database.items():
            for foundation in foundations:
                foundation_lab = [foundation['L'], foundation['a'], foundation['b']]
                predicted_result = self.predict_foundation_match(skin_lab, foundation_lab)
                
                # Calculate color difference (Delta E)
                delta_e = self.calculate_delta_e(skin_lab, predicted_result)
                
                # Apply undertone matching bonus
                undertone_bonus = 0
                if foundation['undertone'] == detected_undertone['primary']:
                    undertone_bonus = -0.5  # Reduce delta_e for matching undertones
                elif foundation['undertone'] == detected_undertone['secondary']:
                    undertone_bonus = -0.25  # Smaller bonus for secondary undertone
                
                adjusted_delta_e = max(0, delta_e + undertone_bonus)
                
                matches.append({
                    'foundation': foundation,
                    'predicted_result': predicted_result,
                    'delta_e': delta_e,
                    'adjusted_delta_e': adjusted_delta_e,
                    'tone_category': tone_category,
                    'undertone_match': foundation['undertone'] == detected_undertone['primary']
                })
        
        # Sort by adjusted Delta E (best match)
        matches.sort(key=lambda x: x['adjusted_delta_e'])
        return matches[:num_matches]
    
    def detect_undertone(self, lab_color):
        """Detect skin undertone from LAB values"""
        L, a, b = lab_color
        
        # Analyze undertone based on a* and b* values
        # a* positive = red, negative = green
        # b* positive = yellow, negative = blue
        
        undertone_info = {
            'primary': 'Neutral',
            'secondary': None,
            'confidence': 0.0,
            'analysis': {}
        }
        
        # Calculate undertone indicators
        warmth_indicator = b / (abs(a) + 1)  # Higher values indicate warmer
        coolness_indicator = a / (abs(b) + 1)  # Lower values indicate cooler
        
        # Determine primary undertone
        if warmth_indicator > 2.5 and a > 5:
            undertone_info['primary'] = 'Warm'
            undertone_info['confidence'] = min(100, warmth_indicator * 20)
        elif warmth_indicator < 1.5 and a < 10:
            undertone_info['primary'] = 'Cool'
            undertone_info['confidence'] = min(100, (2 - warmth_indicator) * 50)
        else:
            undertone_info['primary'] = 'Neutral'
            undertone_info['confidence'] = 100 - abs(warmth_indicator - 2) * 25
        
        # Check for special undertones
        if a > 20 and b < 10:
            undertone_info['secondary'] = 'Red'
        
        # Store detailed analysis
        undertone_info['analysis'] = {
            'warmth_indicator': round(warmth_indicator, 2),
            'coolness_indicator': round(coolness_indicator, 2),
            'a_value': round(a, 2),
            'b_value': round(b, 2)
        }
        
        return undertone_info
    
    def calculate_delta_e(self, lab1, lab2):
        """Calculate Delta E 76 color difference"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))

class SkinDetector:
    """Enhanced skin detection with multiple algorithms and post-processing"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_skin(self, image, use_face_detection=True):
        """Detect skin pixels using enhanced multi-method approach with fair skin optimization"""
        h, w, c = image.shape
        
        # Apply lighting normalization first
        normalized_image = self._normalize_lighting(image)
        
        # Method 1: RGB-H-CbCr (enhanced for fair skin)
        skin_mask_rgbhcbcr = self._detect_skin_rgbhcbcr_enhanced(normalized_image)
        
        # Method 2: YCrCb range-based detection (enhanced ranges)
        skin_mask_ycrcb = self._detect_skin_ycrcb_enhanced(normalized_image)
        
        # Method 3: HSV range-based detection (enhanced ranges)
        skin_mask_hsv = self._detect_skin_hsv_enhanced(normalized_image)
        
        # Method 4: Fair skin specific detection
        skin_mask_fair = self._detect_fair_skin(normalized_image)
        
        # Adaptive voting based on image brightness
        avg_brightness = np.mean(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY))
        
        if avg_brightness > 150:  # Bright image - likely fair skin
            # More lenient voting for fair skin detection
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            vote_count = (skin_mask_rgbhcbcr > 0).astype(int) + \
                         (skin_mask_ycrcb > 0).astype(int) + \
                         (skin_mask_hsv > 0).astype(int) + \
                         (skin_mask_fair > 0).astype(int)
            # At least 2 methods must agree, but fair skin method gets priority
            combined_mask[vote_count >= 2] = 255
            combined_mask[skin_mask_fair > 0] = 255  # Include fair skin detections
        else:
            # Standard voting for darker images
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            vote_count = (skin_mask_rgbhcbcr > 0).astype(int) + \
                         (skin_mask_ycrcb > 0).astype(int) + \
                         (skin_mask_hsv > 0).astype(int)
            combined_mask[vote_count >= 2] = 255
        
        # Apply face detection to focus on facial skin if enabled
        if use_face_detection:
            face_mask = self._create_face_mask(normalized_image)
            if face_mask is not None:
                # Prioritize skin within face region
                combined_mask = cv2.bitwise_and(combined_mask, face_mask)
        
        # Post-processing to remove noise and fill holes
        combined_mask = self._post_process_mask(combined_mask)
        
        return combined_mask
    
    def _normalize_lighting(self, image):
        """Apply lighting normalization to improve skin detection consistency"""
        # Convert to LAB color space for better lighting normalization
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for very dark images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 80:  # Very dark image
            gamma = 1.5  # Brighten
            normalized = np.power(normalized / 255.0, 1.0 / gamma) * 255
            normalized = normalized.astype(np.uint8)
        elif avg_brightness > 200:  # Very bright image
            gamma = 0.8  # Darken slightly
            normalized = np.power(normalized / 255.0, 1.0 / gamma) * 255
            normalized = normalized.astype(np.uint8)
        
        return normalized
    
    def _detect_fair_skin(self, image):
        """Specific detection method optimized for fair skin tones"""
        h, w, c = image.shape
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to different color spaces
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        r = rgb_image[:, :, 0].astype(float)
        g = rgb_image[:, :, 1].astype(float)
        b = rgb_image[:, :, 2].astype(float)
        
        h_val = hsv_image[:, :, 0].astype(float)
        s_val = hsv_image[:, :, 1].astype(float)
        v_val = hsv_image[:, :, 2].astype(float)
        
        cr = ycrcb_image[:, :, 1].astype(float)
        cb = ycrcb_image[:, :, 2].astype(float)
        
        # Fair skin specific conditions
        # 1. High brightness values
        brightness_cond = (r > 180) & (g > 160) & (b > 140)
        
        # 2. Low saturation (fair skin is less saturated)
        saturation_cond = s_val < 60
        
        # 3. Specific hue range for fair skin
        h_degrees = h_val * 2
        hue_cond = ((h_degrees < 30) | (h_degrees > 330)) | ((h_degrees > 10) & (h_degrees < 25))
        
        # 4. RGB ratios typical of fair skin
        ratio_cond = (r > g) & (g > b) & ((r - g) < 40) & ((g - b) < 30)
        
        # 5. CrCb conditions adjusted for fair skin
        cr_cb_cond = (cr > 130) & (cr < 170) & (cb > 100) & (cb < 140)
        
        # Combine conditions (more lenient for fair skin)
        fair_skin_cond = brightness_cond & saturation_cond & (hue_cond | ratio_cond | cr_cb_cond)
        
        skin_mask[fair_skin_cond] = 255
        
        return skin_mask
    
    def _detect_skin_rgbhcbcr_enhanced(self, image):
        """Enhanced RGB-H-CbCr skin detection with fair skin optimization"""
        h, w, c = image.shape
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to YCrCb for Cr and Cb channels
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Convert to HSV for H channel
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Vectorized approach for better performance
        r = rgb_image[:, :, 0].astype(float)
        g = rgb_image[:, :, 1].astype(float)
        b = rgb_image[:, :, 2].astype(float)
        
        y = ycrcb_image[:, :, 0].astype(float)
        cr = ycrcb_image[:, :, 1].astype(float)
        cb = ycrcb_image[:, :, 2].astype(float)
        
        h_val = hsv_image[:, :, 0].astype(float)
        
        # Enhanced Criterion 1: RGB conditions (lowered thresholds for fair skin)
        cond1 = ((r > 80) & (g > 35) & (b > 15) &  # Lowered from 95, 40, 20
                (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) > 10) &  # Lowered from 15
                (np.abs(r - g) > 10) & (r > g) & (r > b))  # Lowered from 15
        
        # Enhanced Criterion 2: CrCb conditions (expanded for fair skin)
        cond2 = ((cr <= 1.5862 * cb + 25) &  # Slightly expanded
                (cr >= 0.3448 * cb + 70) &   # Lowered threshold
                (cr >= -4.5652 * cb + 230) &  # Adjusted
                (cr <= -1.15 * cb + 310) &    # Expanded
                (cr <= -2.2857 * cb + 440))   # Expanded
        
        # Enhanced Criterion 3: Hue conditions (expanded range)
        h_degrees = h_val * 2
        cond3 = (h_degrees < 35) | (h_degrees > 220)  # Expanded from 25/230
        
        skin_mask[cond1 & cond2 & cond3] = 255
        
        return skin_mask
    
    def _detect_skin_ycrcb_enhanced(self, image):
        """Enhanced YCrCb range-based skin detection with fair skin support"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define multiple skin color ranges for different skin tones
        # Range 1: Fair skin
        lower_fair = np.array([0, 125, 70], dtype=np.uint8)
        upper_fair = np.array([255, 180, 135], dtype=np.uint8)
        
        # Range 2: Medium skin (original range)
        lower_medium = np.array([0, 133, 77], dtype=np.uint8)
        upper_medium = np.array([255, 173, 127], dtype=np.uint8)
        
        # Range 3: Dark skin
        lower_dark = np.array([0, 140, 85], dtype=np.uint8)
        upper_dark = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create masks for each range
        mask_fair = cv2.inRange(ycrcb, lower_fair, upper_fair)
        mask_medium = cv2.inRange(ycrcb, lower_medium, upper_medium)
        mask_dark = cv2.inRange(ycrcb, lower_dark, upper_dark)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_fair, mask_medium)
        combined_mask = cv2.bitwise_or(combined_mask, mask_dark)
        
        return combined_mask
    
    def _detect_skin_hsv_enhanced(self, image):
        """Enhanced HSV range-based skin detection with fair skin support"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define multiple HSV ranges for different skin tones
        # Range 1: Fair skin (lower saturation, higher value)
        lower_fair = np.array([0, 10, 120], dtype=np.uint8)
        upper_fair = np.array([25, 80, 255], dtype=np.uint8)
        
        # Range 2: Medium skin (original range)
        lower_medium = np.array([0, 20, 70], dtype=np.uint8)
        upper_medium = np.array([20, 255, 255], dtype=np.uint8)
        
        # Range 3: Additional range for pinkish fair skin
        lower_pink = np.array([160, 10, 100], dtype=np.uint8)
        upper_pink = np.array([179, 60, 255], dtype=np.uint8)
        
        # Create masks for each range
        mask_fair = cv2.inRange(hsv, lower_fair, upper_fair)
        mask_medium = cv2.inRange(hsv, lower_medium, upper_medium)
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_fair, mask_medium)
        combined_mask = cv2.bitwise_or(combined_mask, mask_pink)
        
        return combined_mask
    
    def _detect_skin_rgbhcbcr(self, image):
        """Original RGB-H-CbCr skin detection algorithm"""
        h, w, c = image.shape
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to YCrCb for Cr and Cb channels
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Convert to HSV for H channel
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Vectorized approach for better performance
        r = rgb_image[:, :, 0].astype(float)
        g = rgb_image[:, :, 1].astype(float)
        b = rgb_image[:, :, 2].astype(float)
        
        y = ycrcb_image[:, :, 0].astype(float)
        cr = ycrcb_image[:, :, 1].astype(float)
        cb = ycrcb_image[:, :, 2].astype(float)
        
        h_val = hsv_image[:, :, 0].astype(float)
        
        # Criterion 1: RGB conditions
        cond1 = ((r > 95) & (g > 40) & (b > 20) & 
                (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) > 15) &
                (np.abs(r - g) > 15) & (r > g) & (r > b))
        
        # Criterion 2: CrCb conditions
        cond2 = ((cr <= 1.5862 * cb + 20) & 
                (cr >= 0.3448 * cb + 76.2069) & 
                (cr >= -4.5652 * cb + 234.5652) & 
                (cr <= -1.15 * cb + 301.75) & 
                (cr <= -2.2857 * cb + 432.85))
        
        # Criterion 3: Hue conditions (converting HSV H from 0-179 to 0-359)
        h_degrees = h_val * 2
        cond3 = (h_degrees < 25) | (h_degrees > 230)
        
        skin_mask[cond1 & cond2 & cond3] = 255
        
        return skin_mask
    
    def _detect_skin_ycrcb(self, image):
        """YCrCb range-based skin detection"""
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        return mask
    
    def _detect_skin_hsv(self, image):
        """HSV range-based skin detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        return mask
    
    def _create_face_mask(self, image):
        """Create a mask focusing on face region"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for (x, y, w_face, h_face) in faces:
            # Expand face region slightly to include neck area
            x_expanded = max(0, x - int(w_face * 0.2))
            y_expanded = max(0, y - int(h_face * 0.2))
            w_expanded = min(w - x_expanded, int(w_face * 1.4))
            h_expanded = min(h - y_expanded, int(h_face * 1.4))
            
            # Create elliptical mask for more natural face shape
            center = (x_expanded + w_expanded // 2, y_expanded + h_expanded // 2)
            axes = (w_expanded // 2, h_expanded // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        return mask
    
    def _post_process_mask(self, mask):
        """Post-process mask to remove noise and fill holes"""
        # Remove small noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes with morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component (main skin region)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean = np.zeros_like(mask)
            mask_clean[labels == largest_component] = 255
            return mask_clean
        
        return mask
    
    def get_average_skin_color(self, image, skin_mask):
        """Get average color of skin regions with enhanced statistics for fair skin"""
        skin_pixels = image[skin_mask > 0]
        if len(skin_pixels) == 0:
            return np.array([0, 0, 0])
        
        # For fair skin, use more conservative outlier removal
        # Check if this might be fair skin based on brightness
        avg_brightness = np.mean(skin_pixels)
        
        if avg_brightness > 180:  # Likely fair skin
            # Use more conservative outlier removal for fair skin
            lower_percentile = np.percentile(skin_pixels, 10, axis=0)  # Less aggressive
            upper_percentile = np.percentile(skin_pixels, 90, axis=0)  # Less aggressive
        else:
            # Standard outlier removal for medium/dark skin
            lower_percentile = np.percentile(skin_pixels, 5, axis=0)
            upper_percentile = np.percentile(skin_pixels, 95, axis=0)
        
        # Filter pixels within the percentile range
        mask = np.all((skin_pixels >= lower_percentile) & 
                     (skin_pixels <= upper_percentile), axis=1)
        filtered_pixels = skin_pixels[mask]
        
        if len(filtered_pixels) > 0:
            # Use weighted average based on pixel clustering
            return self._get_weighted_skin_color(filtered_pixels)
        else:
            return np.mean(skin_pixels, axis=0)
    
    def _get_weighted_skin_color(self, skin_pixels):
        """Get weighted average skin color using clustering-based confidence"""
        if len(skin_pixels) < 10:
            return np.median(skin_pixels, axis=0)
        
        try:
            from sklearn.cluster import KMeans
            
            # Use clustering to identify the most representative skin color
            n_clusters = min(3, len(skin_pixels) // 10)
            if n_clusters < 2:
                return np.median(skin_pixels, axis=0)
            
            # Sample for faster processing if too many pixels
            if len(skin_pixels) > 1000:
                sample_indices = np.random.choice(len(skin_pixels), 1000, replace=False)
                sample_pixels = skin_pixels[sample_indices]
            else:
                sample_pixels = skin_pixels
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_pixels)
            
            # Find the largest cluster (most representative color)
            cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
            largest_cluster_idx = np.argmax(cluster_sizes)
            
            # Get pixels from the largest cluster
            main_cluster_pixels = sample_pixels[labels == largest_cluster_idx]
            
            # Return the median of the main cluster for robustness
            return np.median(main_cluster_pixels, axis=0)
            
        except Exception:
            # Fallback to median if clustering fails
            return np.median(skin_pixels, axis=0)
    
    def get_skin_color_statistics(self, image, skin_mask):
        """Get comprehensive skin color statistics"""
        skin_pixels = image[skin_mask > 0]
        if len(skin_pixels) == 0:
            return None
        
        # Convert to LAB for perceptual analysis
        skin_pixels_rgb = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB)
        skin_pixels_lab = cv2.cvtColor(skin_pixels_rgb, cv2.COLOR_RGB2LAB)
        skin_pixels_lab = skin_pixels_lab.reshape(-1, 3)
        
        stats = {
            'mean_rgb': np.mean(skin_pixels, axis=0),
            'median_rgb': np.median(skin_pixels, axis=0),
            'std_rgb': np.std(skin_pixels, axis=0),
            'mean_lab': np.mean(skin_pixels_lab, axis=0),
            'median_lab': np.median(skin_pixels_lab, axis=0),
            'std_lab': np.std(skin_pixels_lab, axis=0),
            'dominant_colors': self._get_dominant_colors(skin_pixels),
            'color_uniformity': self._calculate_color_uniformity(skin_pixels_lab)
        }
        
        return stats
    
    def _get_dominant_colors(self, pixels, n_colors=3):
        """Extract dominant colors using K-means clustering"""
        from sklearn.cluster import KMeans
        
        if len(pixels) < n_colors:
            return pixels.tolist()
        
        # Sample pixels for faster processing
        sample_size = min(1000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(sample_pixels)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_
        
        # Sort by cluster size (most dominant first)
        labels = kmeans.labels_
        cluster_sizes = [np.sum(labels == i) for i in range(n_colors)]
        sorted_indices = np.argsort(cluster_sizes)[::-1]
        
        return dominant_colors[sorted_indices].tolist()
    
    def _calculate_color_uniformity(self, lab_pixels):
        """Calculate how uniform the skin color is (0-1, higher is more uniform)"""
        if len(lab_pixels) < 2:
            return 1.0
        
        # Calculate standard deviation in LAB space
        std_lab = np.std(lab_pixels, axis=0)
        
        # Normalize by typical skin color variation ranges
        # These values are based on typical skin color variations
        typical_std = np.array([10.0, 5.0, 5.0])  # L, a, b
        
        # Calculate uniformity score
        uniformity_scores = 1.0 - np.minimum(std_lab / typical_std, 1.0)
        overall_uniformity = np.mean(uniformity_scores)
        
        return float(overall_uniformity)

# Initialize global calibrator
global_calibrator = ColorCalibrator()

def create_color_swatch(lab_color, size=(100, 50)):
    """Create a color swatch from LAB values with robust color conversion"""
    try:
        L, a, b = lab_color
        
        # Validate LAB values
        L = max(0, min(100, float(L)))
        a = max(-128, min(127, float(a)))
        b = max(-128, min(127, float(b)))
        
        # Method 1: Try professional CIE conversion
        try:
            if hasattr(global_calibrator, 'converter') and global_calibrator.converter:
                rgb_normalized = global_calibrator.converter.lab_to_srgb([L, a, b])
                if rgb_normalized is not None and len(rgb_normalized) == 3:
                    rgb_255 = np.clip(rgb_normalized * 255, 0, 255).astype(np.uint8)
                    color_array = np.full((size[1], size[0], 3), rgb_255)
                    
                    pil_image = Image.fromarray(color_array)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    buffer.seek(0)
                    
                    return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            pass  # Fall through to next method
        
        # Method 2: Standard LAB to XYZ to RGB conversion
        try:
            # LAB to XYZ conversion (D65 illuminant)
            fy = (L + 16) / 116
            fx = a / 500 + fy
            fz = fy - b / 200
            
            # XYZ values (D65 white point)
            xn, yn, zn = 95.047, 100.0, 108.883
            
            def f_inv(t):
                delta = 6/29
                return t**3 if t > delta else 3 * delta**2 * (t - 4/29)
            
            X = xn * f_inv(fx) / 100.0
            Y = yn * f_inv(fy) / 100.0
            Z = zn * f_inv(fz) / 100.0
            
            # XYZ to sRGB conversion matrix
            r = X *  3.2406 + Y * -1.5372 + Z * -0.4986
            g = X * -0.9689 + Y *  1.8758 + Z *  0.0415
            b_rgb = X *  0.0557 + Y * -0.2040 + Z *  1.0570
            
            # Gamma correction for sRGB
            def gamma_correct(c):
                return 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1/2.4)) - 0.055
            
            r = gamma_correct(r)
            g = gamma_correct(g)
            b_rgb = gamma_correct(b_rgb)
            
            # Ensure values are in valid range and convert to 8-bit
            rgb_255 = np.array([r, g, b_rgb])
            rgb_255 = np.clip(rgb_255, 0, 1) * 255
            rgb_255 = rgb_255.astype(np.uint8)
            
            # Create color array for the image
            color_array = np.full((size[1], size[0], 3), rgb_255, dtype=np.uint8)
            
            pil_image = Image.fromarray(color_array, mode='RGB')
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            pass  # Fall through to next method
        
        # Method 3: Simple approximation (most robust fallback)
        L_norm = L / 100.0
        a_norm = a / 128.0  # Normalize a* and b* to [-1, 1]
        b_norm = b / 128.0
        
        # Simple RGB approximation from LAB
        # This isn't colorimetrically accurate but will show something reasonable
        r = L_norm + 0.3 * a_norm + 0.1 * b_norm
        g = L_norm - 0.2 * a_norm + 0.2 * b_norm  
        b_rgb = L_norm - 0.1 * a_norm - 0.4 * b_norm
        
        # Clamp and convert to 8-bit
        rgb_255 = np.clip([r, g, b_rgb], 0, 1) * 255
        rgb_255 = rgb_255.astype(np.uint8)
        
        color_array = np.full((size[1], size[0], 3), rgb_255)
        
        pil_image = Image.fromarray(color_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        # Final fallback: create a neutral gray swatch
        print(f"Error creating color swatch for LAB {lab_color}: {e}")
        
        try:
            # Create a gray swatch based on L value
            L = lab_color[0] if isinstance(lab_color, (list, tuple)) and len(lab_color) > 0 else 50
            gray_value = int(np.clip(L * 2.55, 0, 255))  # Convert L (0-100) to RGB (0-255)
            
            color_array = np.full((size[1], size[0], 3), gray_value, dtype=np.uint8)
            pil_image = Image.fromarray(color_array)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception:
            # Ultimate fallback: return base64 for a simple gray square
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mM8w8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

# Initialize the predictor
predictor = FoundationColorPredictor()

@app.route('/')
def index():
    # Get available brands for the UI
    available_brands = predictor.get_available_brands()
    current_brand = predictor.current_brand
    return render_template('index.html', available_brands=available_brands, current_brand=current_brand)

@app.route('/brands', methods=['GET'])
def get_brands():
    """Get available brands"""
    return jsonify({
        'brands': predictor.get_available_brands(),
        'current': predictor.current_brand
    })

@app.route('/brands/set', methods=['POST'])
def set_brand():
    """Set the current brand for foundation matching"""
    data = request.get_json()
    if not data or 'brand' not in data:
        return jsonify({'error': 'Brand parameter required'}), 400
    
    brand = data['brand']
    if predictor.set_brand(brand):
        # Store in session for persistence
        session['selected_brand'] = brand
        return jsonify({
            'success': True,
            'brand': brand,
            'message': f'Switched to {brand} foundations'
        })
    else:
        return jsonify({'error': f'Invalid brand: {brand}'}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store filename in session
        session['uploaded_file'] = filename
        
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'uploaded_file' not in session:
        return jsonify({'error': 'No file uploaded'}), 400
    
    filename = session['uploaded_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Restore brand from session if available
    if 'selected_brand' in session:
        predictor.set_brand(session['selected_brand'])
    
    try:
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Detect skin
        skin_mask = predictor.skin_detector.detect_skin(image)
        
        # Check if skin was detected
        skin_pixel_count = np.sum(skin_mask > 0)
        if skin_pixel_count == 0:
            return jsonify({'error': 'No skin detected in the image. Please try a different photo with better lighting and a clear view of your face.'}), 400
        
        # Get average skin color (in BGR format from OpenCV)
        avg_skin_bgr = predictor.skin_detector.get_average_skin_color(image, skin_mask)
        
        # Convert BGR to RGB for proper color conversion
        avg_skin_rgb = np.array([avg_skin_bgr[2], avg_skin_bgr[1], avg_skin_bgr[0]])
        
        # Convert to LAB using professional CIE conversion
        if global_calibrator.is_calibrated:
            # Use calibrated conversion for maximum accuracy
            skin_color_lab = global_calibrator.rgb_to_lab_calibrated(avg_skin_rgb).tolist()
        else:
            # Use professional CIE conversion as default (much better than basic conversion)
            skin_color_lab = global_calibrator.converter.srgb_to_lab(avg_skin_rgb).tolist()
        
        # Create skin color swatch
        skin_swatch_b64 = create_color_swatch(skin_color_lab)
        
        # Detect undertone
        undertone_info = predictor.detect_undertone(skin_color_lab)
        
        # Get comprehensive skin statistics
        skin_stats = predictor.skin_detector.get_skin_color_statistics(image, skin_mask)
        
        # Find foundation matches
        matches = predictor.find_best_foundation_matches(skin_color_lab, num_matches=12)
        
        # Prepare matches data with swatches
        matches_data = []
        for i, match in enumerate(matches):
            foundation = match['foundation']
            delta_e = match['delta_e']
            tone_category = match['tone_category']
            
            # Calculate match score (higher is better)
            # Use a more forgiving formula that doesn't drop to 0% as quickly
            # Delta E of 10 = 50% match, Delta E of 20 = 0% match
            match_score = max(0, 100 - (delta_e * 5))
            
            # Create foundation swatch
            foundation_lab = [foundation['L'], foundation['a'], foundation['b']]
            foundation_swatch_b64 = create_color_swatch(foundation_lab)
            
            # Color difference interpretation
            if delta_e < 1:
                match_quality = "Excellent match - barely perceptible difference"
                match_class = "excellent"
            elif delta_e < 2:
                match_quality = "Very good match - slight difference"
                match_class = "very-good"
            elif delta_e < 3:
                match_quality = "Good match - noticeable but acceptable"
                match_class = "good"
            elif delta_e < 5:
                match_quality = "Fair match - clearly noticeable difference"
                match_class = "fair"
            else:
                match_quality = "Poor match - significant difference"
                match_class = "poor"
            
            # Ensure swatch is valid before adding to results
            if not foundation_swatch_b64 or len(foundation_swatch_b64) < 10:
                # Create a simple fallback swatch if needed
                foundation_swatch_b64 = create_color_swatch([50, 0, 0])  # neutral gray
            
            matches_data.append({
                'rank': i + 1,
                'foundation': foundation,
                'delta_e': round(delta_e, 2),
                'tone_category': tone_category,
                'match_score': round(match_score, 1),
                'match_quality': match_quality,
                'match_class': match_class,
                'swatch_b64': foundation_swatch_b64
            })
        
        # Store results in session
        session['analysis_results'] = {
            'skin_color_lab': skin_color_lab,
            'skin_pixel_count': int(skin_pixel_count),
            'skin_swatch_b64': skin_swatch_b64,
            'undertone': undertone_info,
            'skin_stats': {
                'color_uniformity': skin_stats['color_uniformity'] if skin_stats else 0.0
            },
            'matches': matches_data,
            'brand': predictor.current_brand
        }
        
        return jsonify({
            'success': True,
            'skin_color_lab': skin_color_lab,
            'skin_pixel_count': int(skin_pixel_count),
            'skin_swatch_b64': skin_swatch_b64,
            'undertone': undertone_info,
            'skin_stats': {
                'color_uniformity': skin_stats['color_uniformity'] if skin_stats else 0.0
            },
            'matches': matches_data,
            'brand': predictor.current_brand
        })
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing image: {str(e)}'}), 500

@app.route('/results')
def results():
    if 'analysis_results' not in session:
        return redirect('/')
    
    return render_template('results.html', results=session['analysis_results'])

# Calibration routes
@app.route('/calibration')
def calibration():
    """Calibration page for ColorChecker setup"""
    calibration_info = global_calibrator.get_calibration_info()
    return render_template('calibration.html', calibration_info=calibration_info)

@app.route('/calibration/upload', methods=['POST'])
def upload_colorchecker():
    """Upload ColorChecker image for calibration"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = 'colorchecker_' + str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store filename in session
        session['colorchecker_file'] = filename
        
        return jsonify({'success': True, 'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/calibration/detect', methods=['POST'])
def detect_colorchecker():
    """Detect ColorChecker chart in uploaded image"""
    if 'colorchecker_file' not in session:
        return jsonify({'error': 'No ColorChecker file uploaded'}), 400
    
    filename = session['colorchecker_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Detect ColorChecker chart
        chart_data = global_calibrator.detector.detect_chart(image, debug=True)
        
        if chart_data is None:
            return jsonify({
                'success': False,
                'error': 'Could not detect ColorChecker chart automatically',
                'suggestions': [
                    'Ensure the ColorChecker chart is clearly visible',
                    'Check lighting conditions - avoid shadows and reflections',
                    'Make sure the chart fills a good portion of the frame',
                    'Try manual corner selection if automatic detection fails'
                ]
            }), 400
        
        # Validate detection
        validation = global_calibrator.detector.validate_detection(chart_data['patches'])
        
        # Store chart data in session
        session['chart_data'] = {
            'corners': chart_data['corners'].tolist(),
            'patches': chart_data['patches'],
            'validation': validation
        }
        
        return jsonify({
            'success': True,
            'chart_detected': True,
            'quality_score': validation['quality_score'],
            'validation': validation,
            'chart_area': chart_data.get('chart_area', 0),
            'aspect_ratio': chart_data.get('aspect_ratio', 0)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error detecting ColorChecker: {str(e)}'}), 500

@app.route('/calibration/manual', methods=['POST'])
def manual_colorchecker_selection():
    """Manual ColorChecker corner selection"""
    if 'colorchecker_file' not in session:
        return jsonify({'error': 'No ColorChecker file uploaded'}), 400
    
    data = request.get_json()
    if not data or 'corners' not in data:
        return jsonify({'error': 'Corner coordinates required'}), 400
    
    corners = data['corners']
    if len(corners) != 4:
        return jsonify({'error': 'Exactly 4 corners required'}), 400
    
    filename = session['colorchecker_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Manual selection
        chart_data = global_calibrator.detector.manual_selection(image, corners)
        
        if chart_data is None:
            return jsonify({'error': 'Invalid corner selection'}), 400
        
        # Validate detection
        validation = global_calibrator.detector.validate_detection(chart_data['patches'])
        
        # Store chart data in session
        session['chart_data'] = {
            'corners': chart_data['corners'].tolist(),
            'patches': chart_data['patches'],
            'validation': validation,
            'manual_selection': True
        }
        
        return jsonify({
            'success': True,
            'chart_detected': True,
            'quality_score': validation['quality_score'],
            'validation': validation,
            'manual_selection': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Error with manual selection: {str(e)}'}), 500

@app.route('/calibration/calibrate', methods=['POST'])
def perform_calibration():
    """Perform color calibration using detected ColorChecker"""
    if 'colorchecker_file' not in session or 'chart_data' not in session:
        return jsonify({'error': 'No ColorChecker data available'}), 400
    
    filename = session['colorchecker_file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    chart_data = session['chart_data']
    
    try:
        # Load image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Reconstruct chart data
        corners = np.array(chart_data['corners']) if chart_data.get('corners') else None
        manual_corners = corners.tolist() if chart_data.get('manual_selection') else None
        
        # Perform calibration
        calibration_result = global_calibrator.calibrate_from_image(image, manual_corners)
        
        if not calibration_result['success']:
            return jsonify({
                'success': False,
                'error': calibration_result['error'],
                'suggestions': calibration_result.get('suggestions', [])
            }), 400
        
        # Save calibration profile
        calibration_dir = 'static/calibration_profiles'
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_file = os.path.join(calibration_dir, 'current_calibration.json')
        global_calibrator.save_calibration(calibration_file)
        
        return jsonify({
            'success': True,
            'calibrated': True,
            'quality_score': calibration_result['quality_score'],
            'color_accuracy': calibration_result['color_accuracy'],
            'metadata': calibration_result['metadata']
        })
        
    except Exception as e:
        return jsonify({'error': f'Calibration failed: {str(e)}'}), 500

@app.route('/calibration/status')
def calibration_status():
    """Get current calibration status"""
    return jsonify(global_calibrator.get_calibration_info())

@app.route('/calibration/reset', methods=['POST'])
def reset_calibration():
    """Reset calibration to uncalibrated state"""
    global_calibrator.reset_calibration()
    
    # Clear session data
    if 'colorchecker_file' in session:
        del session['colorchecker_file']
    if 'chart_data' in session:
        del session['chart_data']
    
    return jsonify({'success': True, 'calibrated': False})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    # Get port from environment variable for Cloud Run compatibility
    port = int(os.environ.get('PORT', 9090))
    # Disable debug mode in production
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
