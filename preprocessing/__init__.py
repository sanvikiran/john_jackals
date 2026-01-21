"""
Image Preprocessing Pipeline for Eyeglass Prescription Reader

A robust, modular preprocessing pipeline designed for enhancing eyeglass 
prescription images before OCR processing.

Author: Computer Vision Engineer
Date: January 21, 2026
Version: 1.0.0
"""

# Main interface functions
from .preprocess import (
    preprocess_image,
    preprocess_batch,
    get_supported_formats,
    validate_input_file,
    PreprocessingPipeline
)

# Configuration classes
from .config import (
    PreprocessingConfig,
    PresetConfigs,
    validate_config,
    get_config_summary
)

# Utility classes (for advanced users)
from .pdf_utils import PDFProcessor
from .image_utils import ImageProcessor

__version__ = "1.0.0"
__author__ = "Computer Vision Engineer"

# Main exports for easy importing
__all__ = [
    # Core functions
    "preprocess_image",
    "preprocess_batch",
    
    # Configuration
    "PreprocessingConfig", 
    "PresetConfigs",
    
    # Validation and utilities
    "get_supported_formats",
    "validate_input_file",
    
    # Advanced classes
    "PreprocessingPipeline",
    "PDFProcessor", 
    "ImageProcessor",
    
    # Configuration utilities
    "validate_config",
    "get_config_summary"
]

# Quick usage example in docstring
"""
Quick Start:

    from preprocessing import preprocess_image
    
    # Basic usage
    results = preprocess_image("prescription.pdf")
    
    # With configuration
    results = preprocess_image(
        input_path="prescription.jpg",
        preset="handwritten",
        save_outputs=True,
        debug=True
    )
    
    # Each result contains:
    for result in results:
        processed_image = result['image']     # numpy array
        info = result['info']                # processing details
        output_path = result['output_path']  # saved file path
"""