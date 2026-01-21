"""
Main preprocessing pipeline for Eyeglass Prescription Image Reader.

This module provides the main interface for image preprocessing, handling
input format detection, PDF conversion, and orchestrating the OpenCV-based
preprocessing pipeline.

Usage:
    from preprocessing import preprocess_image
    
    # Process a single image or PDF
    processed_images = preprocess_image(
        input_path="prescription.pdf",
        save_outputs=True,
        debug=True
    )

Author: Computer Vision Engineer
Date: January 21, 2026
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
import tempfile

import cv2
import numpy as np
from PIL import Image

from .config import PreprocessingConfig, PresetConfigs, validate_config, get_config_summary
from .pdf_utils import PDFProcessor, validate_pdf_file
from .image_utils import ImageProcessor, save_image_with_info, get_image_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Main preprocessing pipeline class that coordinates all preprocessing steps.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration (uses default if None)
        """
        self.config = config or PreprocessingConfig()
        validate_config(self.config)
        
        # Initialize processors
        self.pdf_processor = PDFProcessor(dpi=self.config.target_dpi)
        self.image_processor = ImageProcessor(self.config)
        
        # Setup logging
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        logger.info("Preprocessing pipeline initialized")
        if self.config.debug_mode:
            logger.info("Debug mode enabled - intermediate steps will be saved")
    
    def process_input(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_outputs: Optional[bool] = None,
        debug: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Process input file (image or PDF) through preprocessing pipeline.
        
        Args:
            input_path: Path to input file (JPEG, PNG, or PDF)
            output_dir: Directory to save processed images (optional)
            save_outputs: Override config save_outputs setting
            debug: Override config debug_mode setting
            
        Returns:
            List of dictionaries containing:
                - 'image': processed image as numpy array
                - 'info': processing information
                - 'original_path': path to original file
                - 'page_number': page number (for PDFs)
                - 'output_path': path to saved file (if saved)
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        # Override config settings if provided
        if save_outputs is not None:
            self.config.save_outputs = save_outputs
        if debug is not None:
            self.config.debug_mode = debug
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Starting preprocessing pipeline for: {input_path}")
        logger.info(f"File size: {input_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent / "outputs" / "processed"
        else:
            output_dir = Path(output_dir)
        
        if self.config.save_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
        
        # Log configuration summary
        if self.config.debug_mode:
            logger.info("Configuration Summary:")
            for line in get_config_summary(self.config).split('\n'):
                logger.info(line)
        
        # Process based on file type
        file_extension = input_path.suffix.lower()
        
        if file_extension == '.pdf':
            results = self._process_pdf(input_path, output_dir)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            results = self._process_image(input_path, output_dir)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Log overall processing stats
        total_time = time.time() - start_time
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        logger.info(f"Processed {len(results)} image(s)")
        
        return results
    
    def _process_pdf(self, pdf_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process PDF file by converting to images first."""
        logger.info("Processing PDF file")
        
        # Validate PDF
        is_valid, message = validate_pdf_file(pdf_path)
        if not is_valid:
            raise ValueError(f"Invalid PDF file: {message}")
        
        # Get PDF info
        pdf_info = self.pdf_processor.get_pdf_info(pdf_path)
        logger.info(f"PDF has {pdf_info['page_count']} pages")
        
        # Convert PDF to images
        try:
            images = self.pdf_processor.convert_pdf_to_images(
                pdf_path,
                save_images=False  # We'll save processed versions
            )
        except Exception as e:
            logger.error(f"Failed to convert PDF: {str(e)}")
            raise
        
        # Process each page
        results = []
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Generate page name
            image_name = f"{pdf_path.stem}_page_{page_num:03d}"
            
            # Process image
            processed_image, processing_info = self.image_processor.process_image(
                image, image_name
            )
            
            # Save debug images if enabled
            if self.config.debug_mode:
                self._save_debug_images(image, processed_image, output_dir, image_name)
            
            # Prepare result
            result = {
                'image': processed_image,
                'info': processing_info,
                'original_path': str(pdf_path),
                'page_number': page_num,
                'output_path': None
            }
            
            # Save final processed image
            if self.config.save_outputs:
                output_filename = self._generate_output_filename(image_name)
                output_path = output_dir / output_filename
                save_image_with_info(processed_image, str(output_path), processing_info, self.config)
                result['output_path'] = str(output_path)
            
            results.append(result)
        
        return results
    
    def _process_image(self, image_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Process single image file."""
        logger.info("Processing image file")
        
        # Load image
        try:
            # Try OpenCV first
            image = cv2.imread(str(image_path))
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
        
        # Log image stats
        if self.config.log_image_stats:
            stats = get_image_stats(image)
            logger.info(f"Original image: {stats['width']}x{stats['height']}, {stats['channels']} channels")
        
        # Process image
        image_name = image_path.stem
        processed_image, processing_info = self.image_processor.process_image(
            image, image_name
        )
        
        # Save debug images if enabled
        if self.config.debug_mode:
            self._save_debug_images(image, processed_image, output_dir, image_name)
        
        # Prepare result
        result = {
            'image': processed_image,
            'info': processing_info,
            'original_path': str(image_path),
            'page_number': 1,
            'output_path': None
        }
        
        # Save final processed image
        if self.config.save_outputs:
            output_filename = self._generate_output_filename(image_name)
            output_path = output_dir / output_filename
            save_image_with_info(processed_image, str(output_path), processing_info, self.config)
            result['output_path'] = str(output_path)
        
        return [result]
    
    def _save_debug_images(
        self, 
        original: np.ndarray, 
        processed: np.ndarray, 
        output_dir: Path, 
        base_name: str
    ):
        """Save debug images showing processing steps."""
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        # Save original
        original_path = debug_dir / f"{base_name}_01_original.png"
        cv2.imwrite(str(original_path), original)
        
        # Save final processed
        final_path = debug_dir / f"{base_name}_99_final.png"
        cv2.imwrite(str(final_path), processed)
        
        logger.debug(f"Debug images saved to {debug_dir}")
    
    def _generate_output_filename(self, base_name: str) -> str:
        """Generate output filename with proper prefix and timestamp."""
        filename = f"{self.config.output_prefix}{base_name}"
        
        if self.config.include_timestamp:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename += f"_{timestamp}"
        
        filename += f".{self.config.output_format}"
        return filename
    
    def get_processing_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate summary of processing results."""
        if not results:
            return "No images processed."
        
        summary_lines = [
            f"Processing Summary for {len(results)} image(s):",
            "=" * 50
        ]
        
        total_time = sum(
            result['info'].get('processing_time_seconds', 0) 
            for result in results
        )
        
        # Overall stats
        summary_lines.extend([
            f"Total processing time: {total_time:.2f}s",
            f"Average time per image: {total_time/len(results):.2f}s",
            ""
        ])
        
        # Per-image details
        for i, result in enumerate(results, 1):
            info = result['info']
            summary_lines.extend([
                f"Image {i}: {info.get('image_name', 'unknown')}",
                f"  Original shape: {info.get('original_shape', 'unknown')}",
                f"  Final shape: {info.get('final_shape', 'unknown')}",
                f"  Processing time: {info.get('processing_time_seconds', 0):.2f}s",
                f"  Steps applied: {', '.join(info.get('steps_applied', []))}",
                ""
            ])
            
            # Add warnings if any
            warnings = info.get('warnings', [])
            if warnings:
                summary_lines.extend([
                    "  Warnings:",
                    *[f"    - {warning}" for warning in warnings],
                    ""
                ])
        
        return "\n".join(summary_lines)


# ========== Main Interface Functions ==========

def preprocess_image(
    input_path: Union[str, Path],
    config: Optional[PreprocessingConfig] = None,
    preset: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_outputs: bool = True,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Main function for preprocessing images or PDFs.
    
    Args:
        input_path: Path to input file (JPEG, PNG, PDF)
        config: Custom preprocessing configuration
        preset: Use predefined configuration ('high_quality_printed', 'handwritten', 'low_quality', 'minimal')
        output_dir: Directory to save processed images
        save_outputs: Whether to save processed images to disk
        debug: Enable debug mode (saves intermediate steps)
        
    Returns:
        List of processing results (one per image/page)
    """
    # Handle preset configurations
    if preset:
        if preset == "high_quality_printed":
            config = PresetConfigs.get_high_quality_printed()
        elif preset == "handwritten":
            config = PresetConfigs.get_handwritten_optimized()
        elif preset == "low_quality":
            config = PresetConfigs.get_low_quality_scanned()
        elif preset == "minimal":
            config = PresetConfigs.get_minimal_processing()
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        logger.info(f"Using preset configuration: {preset}")
    
    # Create pipeline and process
    pipeline = PreprocessingPipeline(config)
    results = pipeline.process_input(
        input_path=input_path,
        output_dir=output_dir,
        save_outputs=save_outputs,
        debug=debug
    )
    
    # Log summary
    summary = pipeline.get_processing_summary(results)
    for line in summary.split('\n'):
        logger.info(line)
    
    return results


def preprocess_batch(
    input_paths: List[Union[str, Path]],
    config: Optional[PreprocessingConfig] = None,
    preset: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_outputs: bool = True,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple files in batch.
    
    Args:
        input_paths: List of input file paths
        config: Preprocessing configuration
        preset: Preset configuration name
        output_dir: Output directory
        save_outputs: Whether to save outputs
        debug: Enable debug mode
        
    Returns:
        List of all processing results
    """
    all_results = []
    
    logger.info(f"Starting batch processing of {len(input_paths)} files")
    
    for i, input_path in enumerate(input_paths, 1):
        logger.info(f"Processing file {i}/{len(input_paths)}: {input_path}")
        
        try:
            results = preprocess_image(
                input_path=input_path,
                config=config,
                preset=preset,
                output_dir=output_dir,
                save_outputs=save_outputs,
                debug=debug
            )
            all_results.extend(results)
            
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {str(e)}")
            # Add error result
            all_results.append({
                'image': None,
                'info': {'error': str(e)},
                'original_path': str(input_path),
                'page_number': None,
                'output_path': None
            })
    
    logger.info(f"Batch processing completed. {len(all_results)} total results")
    return all_results


def get_supported_formats() -> List[str]:
    """Get list of supported input file formats."""
    return ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']


def validate_input_file(input_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate input file format and readability.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Tuple of (is_valid, message)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        return False, "File does not exist"
    
    file_extension = input_path.suffix.lower()
    supported_formats = get_supported_formats()
    
    if file_extension not in supported_formats:
        return False, f"Unsupported format. Supported: {', '.join(supported_formats)}"
    
    if file_extension == '.pdf':
        return validate_pdf_file(input_path)
    else:
        # Try to load image
        try:
            test_image = cv2.imread(str(input_path))
            if test_image is None:
                # Try PIL as backup
                pil_image = Image.open(input_path)
                np.array(pil_image)  # Test conversion
            return True, "Valid image file"
        except Exception as e:
            return False, f"Cannot read image: {str(e)}"


# ========== CLI Support (Optional) ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess images for OCR")
    parser.add_argument("input_path", help="Input file (image or PDF)")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--preset", choices=["high_quality_printed", "handwritten", "low_quality", "minimal"],
                       help="Use preset configuration")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-save", action="store_true", help="Don't save outputs to disk")
    
    args = parser.parse_args()
    
    try:
        results = preprocess_image(
            input_path=args.input_path,
            preset=args.preset,
            output_dir=args.output_dir,
            save_outputs=not args.no_save,
            debug=args.debug
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Processed {len(results)} image(s)")
        
        if args.no_save:
            print("Images were not saved (--no-save flag used)")
        else:
            print(f"Output directory: {args.output_dir or 'auto-generated'}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)