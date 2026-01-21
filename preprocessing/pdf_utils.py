"""
PDF utilities for converting PDF files to images for preprocessing.

This module handles conversion of both single-page and multi-page PDF files
to images that can be processed by the OpenCV-based preprocessing pipeline.

Dependencies:
    - pdf2image: For PDF to image conversion
    - Pillow (PIL): For image handling
    - fitz (PyMuPDF): Alternative PDF library for better performance

Author: Computer Vision Engineer  
Date: January 21, 2026
"""

import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import tempfile

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import (
        PDFInfoNotInstalledError,
        PDFPageCountError,
        PDFSyntaxError
    )
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. PDF processing will be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.info("PyMuPDF not available. Using pdf2image for PDF processing.")

import numpy as np
from PIL import Image


# Configure logging
logger = logging.getLogger(__name__)


class PDFProcessorError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """
    Handles conversion of PDF files to images for preprocessing.
    
    Supports both pdf2image and PyMuPDF backends for flexibility and performance.
    """
    
    def __init__(self, dpi: int = 300, use_pymupdf: bool = True):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for PDF to image conversion
            use_pymupdf: Whether to prefer PyMuPDF over pdf2image
        """
        self.dpi = dpi
        self.use_pymupdf = use_pymupdf and PYMUPDF_AVAILABLE
        
        if not PDF2IMAGE_AVAILABLE and not PYMUPDF_AVAILABLE:
            raise PDFProcessorError(
                "No PDF processing library available. Install pdf2image or PyMuPDF."
            )
    
    def convert_pdf_to_images(
        self, 
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_images: bool = False,
        image_format: str = "PNG"
    ) -> List[np.ndarray]:
        """
        Convert PDF to list of images (numpy arrays).
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (optional)
            save_images: Whether to save images to disk
            image_format: Image format for saving ("PNG", "JPEG")
            
        Returns:
            List of images as numpy arrays
            
        Raises:
            PDFProcessorError: If PDF conversion fails
            FileNotFoundError: If PDF file doesn't exist
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        logger.info(f"Using {'PyMuPDF' if self.use_pymupdf else 'pdf2image'}")
        
        try:
            if self.use_pymupdf:
                images = self._convert_with_pymupdf(pdf_path)
            else:
                images = self._convert_with_pdf2image(pdf_path)
            
            logger.info(f"Successfully converted {len(images)} pages from PDF")
            
            # Save images to disk if requested
            if save_images and output_dir:
                self._save_images(images, output_dir, pdf_path.stem, image_format)
            
            return images
            
        except Exception as e:
            raise PDFProcessorError(f"Failed to convert PDF {pdf_path}: {str(e)}")
    
    def _convert_with_pymupdf(self, pdf_path: Path) -> List[np.ndarray]:
        """Convert PDF using PyMuPDF (faster, better memory management)."""
        images = []
        
        doc = fitz.open(str(pdf_path))
        
        try:
            for page_num in range(doc.page_count):
                logger.debug(f"Processing page {page_num + 1}/{doc.page_count}")
                
                page = doc.load_page(page_num)
                
                # Calculate zoom factor for desired DPI
                # PyMuPDF default is 72 DPI
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(BytesIO(img_data))
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                images.append(img_array)
                
                # Clean up
                pix = None
                
        finally:
            doc.close()
        
        return images
    
    def _convert_with_pdf2image(self, pdf_path: Path) -> List[np.ndarray]:
        """Convert PDF using pdf2image (more compatible, requires poppler)."""
        try:
            # Convert PDF to PIL Images
            pil_images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt='RGB'
            )
            
            # Convert PIL Images to numpy arrays
            images = []
            for i, pil_img in enumerate(pil_images):
                logger.debug(f"Processing page {i + 1}/{len(pil_images)}")
                img_array = np.array(pil_img)
                images.append(img_array)
            
            return images
            
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            raise PDFProcessorError(f"pdf2image error: {str(e)}")
    
    def _save_images(
        self, 
        images: List[np.ndarray], 
        output_dir: Union[str, Path],
        base_name: str,
        image_format: str
    ):
        """Save converted images to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_array in enumerate(images):
            # Convert numpy array back to PIL Image
            pil_image = Image.fromarray(img_array)
            
            # Generate filename
            page_num = i + 1
            ext = image_format.lower()
            if ext == 'jpeg':
                ext = 'jpg'
            
            filename = f"{base_name}_page_{page_num:03d}.{ext}"
            filepath = output_dir / filename
            
            # Save image
            pil_image.save(filepath, format=image_format.upper())
            logger.debug(f"Saved page {page_num} to {filepath}")
    
    def get_pdf_info(self, pdf_path: Union[str, Path]) -> dict:
        """
        Get information about PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            dict: PDF information (page count, etc.)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        info = {
            'file_path': str(pdf_path),
            'file_size_mb': pdf_path.stat().st_size / (1024 * 1024),
            'page_count': 0
        }
        
        try:
            if self.use_pymupdf and PYMUPDF_AVAILABLE:
                doc = fitz.open(str(pdf_path))
                info['page_count'] = doc.page_count
                doc.close()
            elif PDF2IMAGE_AVAILABLE:
                # Get page count without converting
                from pdf2image import pdfinfo_from_path
                pdf_info = pdfinfo_from_path(str(pdf_path))
                info['page_count'] = pdf_info['Pages']
            else:
                raise PDFProcessorError("No PDF library available")
                
        except Exception as e:
            logger.warning(f"Could not get PDF info: {str(e)}")
            info['error'] = str(e)
        
        return info
    
    def is_pdf_valid(self, pdf_path: Union[str, Path]) -> bool:
        """
        Check if PDF file is valid and readable.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            bool: True if PDF is valid, False otherwise
        """
        try:
            info = self.get_pdf_info(pdf_path)
            return info['page_count'] > 0
        except Exception:
            return False


# ========== Convenience Functions ==========

def pdf_to_images(
    pdf_path: Union[str, Path],
    dpi: int = 300,
    output_dir: Optional[Union[str, Path]] = None,
    save_images: bool = False
) -> List[np.ndarray]:
    """
    Quick function to convert PDF to images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion
        output_dir: Directory to save images (if save_images=True)
        save_images: Whether to save images to disk
        
    Returns:
        List of images as numpy arrays
    """
    processor = PDFProcessor(dpi=dpi)
    return processor.convert_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir, 
        save_images=save_images
    )


def get_pdf_page_count(pdf_path: Union[str, Path]) -> int:
    """
    Get number of pages in PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        int: Number of pages
    """
    processor = PDFProcessor()
    info = processor.get_pdf_info(pdf_path)
    return info.get('page_count', 0)


def validate_pdf_file(pdf_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate PDF file and return status.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return False, "File does not exist"
        
        if not pdf_path.suffix.lower() == '.pdf':
            return False, "File is not a PDF"
        
        processor = PDFProcessor()
        if not processor.is_pdf_valid(pdf_path):
            return False, "PDF file is corrupted or unreadable"
        
        info = processor.get_pdf_info(pdf_path)
        page_count = info.get('page_count', 0)
        
        if page_count == 0:
            return False, "PDF has no pages"
        
        return True, f"Valid PDF with {page_count} pages"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ========== Missing import fix ==========
from io import BytesIO