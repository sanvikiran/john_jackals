"""
Configuration module for image preprocessing pipeline.

This module contains all configurable parameters for the preprocessing steps.
Modify these values to fine-tune the preprocessing behavior for different types
of prescription images (printed vs handwritten, different quality levels, etc.).

Author: Computer Vision Engineer
Date: January 21, 2026
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PreprocessingConfig:
    """Configuration class for all preprocessing parameters."""
    
    # ========== General Settings ==========
    target_dpi: int = 300  # Standard DPI for OCR (300 DPI equivalent)
    debug_mode: bool = False  # Save intermediate processing steps
    save_outputs: bool = True  # Save final processed images
    
    # ========== Pipeline Step Controls ==========
    # Toggle preprocessing steps on/off
    enable_grayscale: bool = True
    enable_noise_removal: bool = True
    enable_contrast_enhancement: bool = True
    enable_thresholding: bool = True
    enable_deskewing: bool = True
    enable_resize: bool = True
    enable_morphological: bool = False  # Optional: can strengthen text
    enable_shadow_removal: bool = False  # Optional: experimental
    enable_border_crop: bool = False  # Optional: remove borders
    
    # ========== Noise Removal Settings ==========
    # Gaussian Blur
    gaussian_kernel_size: Tuple[int, int] = (3, 3)
    gaussian_sigma_x: float = 0.0  # Auto-calculate if 0
    gaussian_sigma_y: float = 0.0  # Auto-calculate if 0
    
    # Median Filter
    median_kernel_size: int = 3  # Must be odd
    
    # Bilateral Filter (edge-preserving)
    bilateral_d: int = 9  # Diameter of pixel neighborhood
    bilateral_sigma_color: float = 75.0  # Filter sigma in color space
    bilateral_sigma_space: float = 75.0  # Filter sigma in coordinate space
    
    # Noise removal method selection
    noise_removal_method: str = "bilateral"  # Options: "gaussian", "median", "bilateral"
    
    # ========== Contrast Enhancement (CLAHE) ==========
    clahe_clip_limit: float = 3.0  # Threshold for contrast limiting
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)  # Size of grid for histogram equalization
    
    # ========== Thresholding Settings ==========
    # Adaptive Thresholding (for handwritten text)
    adaptive_max_value: int = 255
    adaptive_method: int = 1  # cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    adaptive_threshold_type: int = 1  # cv2.THRESH_BINARY
    adaptive_block_size: int = 11  # Size of neighborhood area (must be odd)
    adaptive_c_constant: float = 2.0  # Constant subtracted from mean
    
    # OTSU Thresholding (for printed text)
    otsu_threshold_type: int = 1  # cv2.THRESH_BINARY
    
    # Thresholding method selection
    thresholding_method: str = "adaptive"  # Options: "adaptive", "otsu", "auto"
    
    # Auto detection parameters (for choosing between adaptive/otsu)
    text_type_detection_threshold: float = 0.15  # Threshold for printed vs handwritten detection
    
    # ========== Deskewing Settings ==========
    # Hough Transform parameters
    hough_rho: float = 1.0  # Distance resolution in pixels
    hough_theta_step: float = 0.01745329252  # Angle resolution (1 degree in radians)
    hough_threshold: int = 100  # Accumulator threshold for line detection
    hough_min_line_length: int = 100  # Minimum line length
    hough_max_line_gap: int = 20  # Maximum gap between line segments
    
    # Angle correction limits
    max_skew_angle: float = 45.0  # Maximum angle to correct (degrees)
    min_skew_confidence: float = 0.1  # Minimum confidence to apply correction
    
    # ========== Resize Settings ==========
    # Target size calculation method
    resize_method: str = "dpi_based"  # Options: "dpi_based", "fixed_size", "scale_factor"
    
    # Fixed size settings (when resize_method = "fixed_size")
    target_width: Optional[int] = None  # Set to desired width or None for auto
    target_height: Optional[int] = None  # Set to desired height or None for auto
    maintain_aspect_ratio: bool = True
    
    # Scale factor settings (when resize_method = "scale_factor")
    scale_factor: float = 1.0  # Scaling factor (1.0 = no change)
    
    # DPI-based settings (when resize_method = "dpi_based")
    assume_original_dpi: int = 150  # Assumed DPI if not detectable
    min_dimension: int = 1000  # Minimum width/height after resize
    max_dimension: int = 4000  # Maximum width/height after resize
    
    # ========== Morphological Operations ==========
    # Morphological operations to strengthen text
    morph_operation: str = "close"  # Options: "close", "open", "gradient"
    morph_kernel_shape: str = "rectangle"  # Options: "rectangle", "ellipse", "cross"
    morph_kernel_size: Tuple[int, int] = (2, 2)
    morph_iterations: int = 1
    
    # ========== Shadow Removal ==========
    # Experimental shadow removal (may affect text quality)
    shadow_dilate_kernel_size: Tuple[int, int] = (7, 7)
    shadow_blur_kernel_size: Tuple[int, int] = (21, 21)
    shadow_normalize_method: str = "division"  # Options: "division", "subtraction"
    
    # ========== Border Cropping ==========
    # Automatic border detection and removal
    border_crop_threshold: int = 240  # Threshold for detecting borders (white regions)
    border_crop_margin: int = 10  # Margin to keep around content (pixels)
    min_content_ratio: float = 0.1  # Minimum content ratio to preserve
    
    # ========== Output Settings ==========
    output_format: str = "png"  # Options: "png", "jpg", "tiff"
    output_quality: int = 95  # JPEG quality (0-100)
    output_compression: int = 6  # PNG compression level (0-9)
    
    # File naming
    output_prefix: str = "processed_"
    include_timestamp: bool = False
    
    # ========== Logging Settings ==========
    log_level: str = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
    log_processing_time: bool = True
    log_image_stats: bool = True  # Log image dimensions, channels, etc.


# ========== Predefined Configurations ==========

class PresetConfigs:
    """Predefined configuration presets for common use cases."""
    
    @staticmethod
    def get_high_quality_printed() -> PreprocessingConfig:
        """Configuration optimized for high-quality printed prescriptions."""
        config = PreprocessingConfig()
        config.thresholding_method = "otsu"
        config.noise_removal_method = "gaussian"
        config.enable_morphological = False
        config.clahe_clip_limit = 2.0
        return config
    
    @staticmethod
    def get_handwritten_optimized() -> PreprocessingConfig:
        """Configuration optimized for handwritten prescriptions."""
        config = PreprocessingConfig()
        config.thresholding_method = "adaptive"
        config.noise_removal_method = "bilateral"
        config.enable_morphological = True
        config.clahe_clip_limit = 4.0
        config.adaptive_block_size = 15
        config.adaptive_c_constant = 3.0
        return config
    
    @staticmethod
    def get_low_quality_scanned() -> PreprocessingConfig:
        """Configuration optimized for low-quality or noisy scanned documents."""
        config = PreprocessingConfig()
        config.enable_noise_removal = True
        config.noise_removal_method = "bilateral"
        config.enable_contrast_enhancement = True
        config.enable_morphological = True
        config.clahe_clip_limit = 5.0
        config.bilateral_sigma_color = 100.0
        config.bilateral_sigma_space = 100.0
        return config
    
    @staticmethod
    def get_minimal_processing() -> PreprocessingConfig:
        """Minimal processing for already clean images."""
        config = PreprocessingConfig()
        config.enable_noise_removal = False
        config.enable_contrast_enhancement = False
        config.enable_morphological = False
        config.enable_shadow_removal = False
        config.enable_border_crop = False
        return config


# ========== Helper Functions ==========

def validate_config(config: PreprocessingConfig) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: PreprocessingConfig instance to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If critical parameters are invalid
    """
    # Validate kernel sizes (must be odd)
    if config.median_kernel_size % 2 == 0:
        raise ValueError("median_kernel_size must be odd")
    
    if config.adaptive_block_size % 2 == 0:
        raise ValueError("adaptive_block_size must be odd")
    
    # Validate ranges
    if config.target_dpi <= 0:
        raise ValueError("target_dpi must be positive")
    
    if config.clahe_clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be positive")
    
    if not 0 <= config.output_quality <= 100:
        raise ValueError("output_quality must be between 0 and 100")
    
    if not 0 <= config.output_compression <= 9:
        raise ValueError("output_compression must be between 0 and 9")
    
    # Validate method selections
    valid_noise_methods = {"gaussian", "median", "bilateral"}
    if config.noise_removal_method not in valid_noise_methods:
        raise ValueError(f"noise_removal_method must be one of {valid_noise_methods}")
    
    valid_threshold_methods = {"adaptive", "otsu", "auto"}
    if config.thresholding_method not in valid_threshold_methods:
        raise ValueError(f"thresholding_method must be one of {valid_threshold_methods}")
    
    valid_resize_methods = {"dpi_based", "fixed_size", "scale_factor"}
    if config.resize_method not in valid_resize_methods:
        raise ValueError(f"resize_method must be one of {valid_resize_methods}")
    
    return True


def get_config_summary(config: PreprocessingConfig) -> str:
    """
    Generate a human-readable summary of the configuration.
    
    Args:
        config: PreprocessingConfig instance
        
    Returns:
        str: Formatted configuration summary
    """
    summary_lines = [
        "Preprocessing Configuration Summary:",
        "=" * 40,
        f"Target DPI: {config.target_dpi}",
        f"Debug Mode: {config.debug_mode}",
        f"Save Outputs: {config.save_outputs}",
        "",
        "Enabled Steps:",
        f"  Grayscale: {config.enable_grayscale}",
        f"  Noise Removal: {config.enable_noise_removal} ({config.noise_removal_method})",
        f"  Contrast Enhancement: {config.enable_contrast_enhancement}",
        f"  Thresholding: {config.enable_thresholding} ({config.thresholding_method})",
        f"  Deskewing: {config.enable_deskewing}",
        f"  Resize: {config.enable_resize} ({config.resize_method})",
        f"  Morphological: {config.enable_morphological}",
        f"  Shadow Removal: {config.enable_shadow_removal}",
        f"  Border Crop: {config.enable_border_crop}",
    ]
    
    return "\n".join(summary_lines)