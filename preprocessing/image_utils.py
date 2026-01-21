"""
Image utilities for OpenCV-based preprocessing operations.

This module contains all the OpenCV functions for image preprocessing:
- Grayscale conversion
- Noise removal (Gaussian, Median, Bilateral filtering)
- Contrast enhancement (CLAHE)
- Thresholding (Adaptive, OTSU)
- Deskewing (Hough transform, minAreaRect)
- Resizing and DPI normalization
- Morphological operations
- Shadow removal
- Border cropping

Author: Computer Vision Engineer
Date: January 21, 2026
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Union, List
from math import degrees, radians, sin, cos
import time

try:
    from .config import PreprocessingConfig
except ImportError:
    from config import PreprocessingConfig

# Configure logging
logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""
    pass


class ImageProcessor:
    """
    Handles all OpenCV-based image preprocessing operations.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize image processor with configuration.
        
        Args:
            config: PreprocessingConfig instance
        """
        self.config = config
        self.processing_stats = {}
    
    def process_image(self, image: np.ndarray, image_name: str = "image") -> Tuple[np.ndarray, dict]:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Input image as numpy array
            image_name: Name for logging and debugging
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        start_time = time.time()
        current_image = image.copy()
        
        processing_info = {
            'image_name': image_name,
            'original_shape': image.shape,
            'steps_applied': [],
            'processing_time_seconds': 0.0,
            'warnings': []
        }
        
        logger.info(f"Starting preprocessing pipeline for {image_name}")
        logger.info(f"Original image shape: {image.shape}")
        
        try:
            # Step 1: Convert to grayscale
            if self.config.enable_grayscale and len(current_image.shape) == 3:
                current_image = self.convert_to_grayscale(current_image)
                processing_info['steps_applied'].append('grayscale')
                logger.debug("Applied grayscale conversion")
            
            # Step 2: Noise removal
            if self.config.enable_noise_removal:
                current_image = self.remove_noise(current_image)
                processing_info['steps_applied'].append(f'noise_removal_{self.config.noise_removal_method}')
                logger.debug(f"Applied {self.config.noise_removal_method} noise removal")
            
            # Step 3: Contrast enhancement
            if self.config.enable_contrast_enhancement:
                current_image = self.enhance_contrast(current_image)
                processing_info['steps_applied'].append('contrast_enhancement')
                logger.debug("Applied CLAHE contrast enhancement")
            
            # Step 4: Deskewing (before thresholding for better line detection)
            if self.config.enable_deskewing:
                current_image, skew_angle = self.deskew_image(current_image)
                processing_info['steps_applied'].append('deskewing')
                processing_info['skew_angle_degrees'] = skew_angle
                logger.debug(f"Applied deskewing, angle: {skew_angle:.2f}°")
            
            # Step 5: Thresholding
            if self.config.enable_thresholding:
                current_image = self.apply_thresholding(current_image)
                processing_info['steps_applied'].append(f'thresholding_{self.config.thresholding_method}')
                logger.debug(f"Applied {self.config.thresholding_method} thresholding")
            
            # Step 6: Morphological operations
            if self.config.enable_morphological:
                current_image = self.apply_morphological_operations(current_image)
                processing_info['steps_applied'].append('morphological')
                logger.debug("Applied morphological operations")
            
            # Step 7: Shadow removal (experimental)
            if self.config.enable_shadow_removal:
                current_image = self.remove_shadows(current_image)
                processing_info['steps_applied'].append('shadow_removal')
                processing_info['warnings'].append("Shadow removal is experimental and may affect text quality")
                logger.debug("Applied shadow removal (experimental)")
            
            # Step 8: Border cropping
            if self.config.enable_border_crop:
                current_image = self.crop_borders(current_image)
                processing_info['steps_applied'].append('border_crop')
                logger.debug("Applied border cropping")
            
            # Step 9: Resize/DPI normalization
            if self.config.enable_resize:
                current_image = self.resize_image(current_image)
                processing_info['steps_applied'].append(f'resize_{self.config.resize_method}')
                logger.debug(f"Applied {self.config.resize_method} resizing")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_info['processing_time_seconds'] = processing_time
            processing_info['final_shape'] = current_image.shape
            
            logger.info(f"Preprocessing completed in {processing_time:.2f}s")
            logger.info(f"Final image shape: {current_image.shape}")
            
            return current_image, processing_info
            
        except Exception as e:
            processing_info['error'] = str(e)
            logger.error(f"Preprocessing failed for {image_name}: {str(e)}")
            raise ImageProcessingError(f"Failed to process {image_name}: {str(e)}")
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 2:
            return image  # Already grayscale
        
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using the configured method."""
        if self.config.noise_removal_method == "gaussian":
            return cv2.GaussianBlur(
                image,
                self.config.gaussian_kernel_size,
                self.config.gaussian_sigma_x,
                sigmaY=self.config.gaussian_sigma_y
            )
        
        elif self.config.noise_removal_method == "median":
            return cv2.medianBlur(image, self.config.median_kernel_size)
        
        elif self.config.noise_removal_method == "bilateral":
            return cv2.bilateralFilter(
                image,
                self.config.bilateral_d,
                self.config.bilateral_sigma_color,
                self.config.bilateral_sigma_space
            )
        
        else:
            raise ValueError(f"Unknown noise removal method: {self.config.noise_removal_method}")
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size
        )
        return clahe.apply(image)
    
    def apply_thresholding(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding using the configured method."""
        if self.config.thresholding_method == "adaptive":
            return cv2.adaptiveThreshold(
                image,
                self.config.adaptive_max_value,
                self.config.adaptive_method,
                self.config.adaptive_threshold_type,
                self.config.adaptive_block_size,
                self.config.adaptive_c_constant
            )
        
        elif self.config.thresholding_method == "otsu":
            _, thresh = cv2.threshold(
                image, 0, 255, 
                self.config.otsu_threshold_type + cv2.THRESH_OTSU
            )
            return thresh
        
        elif self.config.thresholding_method == "auto":
            # Automatically choose between adaptive and OTSU based on image characteristics
            if self._is_handwritten_text(image):
                return cv2.adaptiveThreshold(
                    image,
                    self.config.adaptive_max_value,
                    self.config.adaptive_method,
                    self.config.adaptive_threshold_type,
                    self.config.adaptive_block_size,
                    self.config.adaptive_c_constant
                )
            else:
                _, thresh = cv2.threshold(
                    image, 0, 255,
                    self.config.otsu_threshold_type + cv2.THRESH_OTSU
                )
                return thresh
        
        else:
            raise ValueError(f"Unknown thresholding method: {self.config.thresholding_method}")
    
    def deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Deskew image using Hough line transform.
        
        Returns:
            Tuple of (deskewed_image, skew_angle_degrees)
        """
        # Make a copy for edge detection
        if len(image.shape) == 3:
            edges = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            edges = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(edges, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(
            edges,
            self.config.hough_rho,
            self.config.hough_theta_step,
            self.config.hough_threshold
        )
        
        if lines is None:
            logger.warning("No lines detected for deskewing")
            return image, 0.0
        
        # Calculate angles of detected lines
        angles = []
        for rho, theta in lines[:, 0]:
            angle = degrees(theta) - 90  # Convert to skew angle
            if abs(angle) <= self.config.max_skew_angle:
                angles.append(angle)
        
        if not angles:
            logger.warning("No valid angles found for deskewing")
            return image, 0.0
        
        # Calculate median angle (more robust than mean)
        skew_angle = np.median(angles)
        
        # Only apply correction if angle is significant
        if abs(skew_angle) < 0.5:  # Less than 0.5 degrees
            logger.debug(f"Skew angle too small: {skew_angle:.2f}°")
            return image, skew_angle
        
        # Apply rotation
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Apply rotation
        deskewed = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return deskewed, skew_angle
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image based on configured method."""
        h, w = image.shape[:2]
        
        if self.config.resize_method == "dpi_based":
            # Calculate scale factor to achieve target DPI
            current_dpi = self.config.assume_original_dpi
            scale_factor = self.config.target_dpi / current_dpi
            
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            # Apply dimension constraints
            if new_w < self.config.min_dimension or new_h < self.config.min_dimension:
                min_scale = self.config.min_dimension / min(w, h)
                new_w = int(w * min_scale)
                new_h = int(h * min_scale)
            
            if new_w > self.config.max_dimension or new_h > self.config.max_dimension:
                max_scale = self.config.max_dimension / max(w, h)
                new_w = int(w * max_scale)
                new_h = int(h * max_scale)
        
        elif self.config.resize_method == "fixed_size":
            if self.config.target_width and self.config.target_height:
                if self.config.maintain_aspect_ratio:
                    # Scale to fit within target dimensions
                    scale_w = self.config.target_width / w
                    scale_h = self.config.target_height / h
                    scale = min(scale_w, scale_h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                else:
                    new_w = self.config.target_width
                    new_h = self.config.target_height
            else:
                # Use original size if target not specified
                return image
        
        elif self.config.resize_method == "scale_factor":
            new_w = int(w * self.config.scale_factor)
            new_h = int(h * self.config.scale_factor)
        
        else:
            raise ValueError(f"Unknown resize method: {self.config.resize_method}")
        
        # Perform resize
        if new_w != w or new_h != h:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Resized from {w}x{h} to {new_w}x{new_h}")
            return resized
        
        return image
    
    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to strengthen text."""
        # Create kernel
        if self.config.morph_kernel_shape == "rectangle":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.morph_kernel_size)
        elif self.config.morph_kernel_shape == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.morph_kernel_size)
        elif self.config.morph_kernel_shape == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, self.config.morph_kernel_size)
        else:
            raise ValueError(f"Unknown kernel shape: {self.config.morph_kernel_shape}")
        
        # Apply operation
        if self.config.morph_operation == "close":
            result = cv2.morphologyEx(
                image, cv2.MORPH_CLOSE, kernel, 
                iterations=self.config.morph_iterations
            )
        elif self.config.morph_operation == "open":
            result = cv2.morphologyEx(
                image, cv2.MORPH_OPEN, kernel,
                iterations=self.config.morph_iterations
            )
        elif self.config.morph_operation == "gradient":
            result = cv2.morphologyEx(
                image, cv2.MORPH_GRADIENT, kernel,
                iterations=self.config.morph_iterations
            )
        else:
            raise ValueError(f"Unknown morphological operation: {self.config.morph_operation}")
        
        return result
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Experimental shadow removal.
        Warning: May affect text quality.
        """
        # Create shadow mask by dilating the image
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.config.shadow_dilate_kernel_size
        )
        dilated = cv2.dilate(image, kernel, iterations=1)
        
        # Blur the dilated image to create background estimate
        bg = cv2.medianBlur(dilated, self.config.shadow_blur_kernel_size[0])
        
        # Normalize using background
        if self.config.shadow_normalize_method == "division":
            # Avoid division by zero
            bg_float = bg.astype(np.float32) + 1
            img_float = image.astype(np.float32)
            result = np.clip((img_float / bg_float) * 255, 0, 255).astype(np.uint8)
        elif self.config.shadow_normalize_method == "subtraction":
            result = cv2.subtract(bg, image)
        else:
            raise ValueError(f"Unknown shadow normalization method: {self.config.shadow_normalize_method}")
        
        return result
    
    def crop_borders(self, image: np.ndarray) -> np.ndarray:
        """Automatically crop white borders from image."""
        # Find content boundaries
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to find non-white regions
        _, binary = cv2.threshold(gray, self.config.border_crop_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No content found for border cropping")
            return image
        
        # Get bounding box of all content
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add margin
        margin = self.config.border_crop_margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)
        
        # Check if crop area is reasonable
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        original_area = image.shape[0] * image.shape[1]
        crop_area = crop_width * crop_height
        
        content_ratio = crop_area / original_area
        if content_ratio < self.config.min_content_ratio:
            logger.warning(f"Crop area too small ({content_ratio:.2%}), skipping border crop")
            return image
        
        # Perform crop
        if len(image.shape) == 3:
            cropped = image[y_min:y_max, x_min:x_max, :]
        else:
            cropped = image[y_min:y_max, x_min:x_max]
        
        logger.debug(f"Cropped from {image.shape} to {cropped.shape}")
        return cropped
    
    def _is_handwritten_text(self, image: np.ndarray) -> bool:
        """
        Heuristic to detect if image contains handwritten text.
        
        This is a simple heuristic based on edge density and could be improved
        with machine learning approaches.
        """
        # Calculate edge density
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Handwritten text typically has more irregular edges
        return edge_density > self.config.text_type_detection_threshold


# ========== Convenience Functions ==========

def preprocess_single_image(
    image: np.ndarray, 
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, dict]:
    """
    Quick function to preprocess a single image.
    
    Args:
        image: Input image as numpy array
        config: Preprocessing configuration (uses default if None)
        
    Returns:
        Tuple of (processed_image, processing_info)
    """
    if config is None:
        config = PreprocessingConfig()
    
    processor = ImageProcessor(config)
    return processor.process_image(image)


def save_image_with_info(
    image: np.ndarray, 
    filepath: str, 
    processing_info: dict,
    config: PreprocessingConfig
):
    """
    Save processed image with processing information.
    
    Args:
        image: Processed image
        filepath: Output file path
        processing_info: Processing information dict
        config: Configuration used
    """
    # Save image
    if config.output_format.lower() == 'jpg' or config.output_format.lower() == 'jpeg':
        cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, config.output_quality])
    elif config.output_format.lower() == 'png':
        cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, config.output_compression])
    else:
        cv2.imwrite(filepath, image)
    
    # Log processing info
    logger.info(f"Saved processed image: {filepath}")
    logger.info(f"Steps applied: {', '.join(processing_info.get('steps_applied', []))}")
    if 'processing_time_seconds' in processing_info:
        logger.info(f"Processing time: {processing_info['processing_time_seconds']:.2f}s")


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get basic statistics about an image.
    
    Args:
        image: Image as numpy array
        
    Returns:
        dict: Image statistics
    """
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'channels': len(image.shape),
        'min_value': float(np.min(image)),
        'max_value': float(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    }
    
    if len(image.shape) == 2:
        stats['width'] = image.shape[1]
        stats['height'] = image.shape[0]
    elif len(image.shape) == 3:
        stats['width'] = image.shape[1]
        stats['height'] = image.shape[0]
        stats['channels'] = image.shape[2]
    
    return stats