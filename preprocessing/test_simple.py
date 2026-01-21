#!/usr/bin/env python3
"""
Simple test script to demonstrate preprocessing pipeline.
This script works around import issues by using absolute imports.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import modules with absolute imports
from config import PreprocessingConfig
from image_utils import ImageProcessor
from pdf_utils import PDFProcessor


def simple_preprocess_test():
    """Simple preprocessing test function."""
    
    print("🔬 Simple Preprocessing Test")
    print("=" * 40)
    
    # Check if test image exists
    input_path = Path("examples/input/test_prescription.jpg")
    if not input_path.exists():
        print("❌ Test image not found!")
        return
    
    print(f"📸 Input: {input_path}")
    
    # Load image
    image = cv2.imread(str(input_path))
    if image is None:
        print("❌ Could not load image!")
        return
    
    print(f"📐 Original size: {image.shape}")
    
    # Create configuration
    config = PreprocessingConfig()
    config.debug_mode = True
    config.save_outputs = True
    
    # Create processor and process image
    processor = ImageProcessor(config)
    
    try:
        processed_image, info = processor.process_image(image, "test_prescription")
        
        # Create output directory
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        
        # Save processed image
        output_path = output_dir / "processed_test_prescription.png"
        cv2.imwrite(str(output_path), processed_image)
        
        # Results
        print("✅ SUCCESS!")
        print(f"📐 Final size: {processed_image.shape}")
        print(f"⏱️  Processing time: {info['processing_time_seconds']:.2f}s")
        print(f"🛠️  Steps applied: {', '.join(info['steps_applied'])}")
        print(f"💾 Output saved: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    simple_preprocess_test()