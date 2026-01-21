# Image Preprocessing Pipeline for Eyeglass Prescription Reader

A robust, modular image preprocessing pipeline designed specifically for enhancing eyeglass prescription images before OCR processing. This module handles both printed and handwritten prescriptions with configurable OpenCV-based preprocessing steps.

## 🎯 Overview

This preprocessing module is part of a larger Eyeglass Prescription Image Reader system. It focuses exclusively on image quality enhancement to improve OCR accuracy, without implementing any OCR functionality itself.

```
Input (JPEG / PNG / PDF) → [ This Module ] → OCR Engine → Text Processing → LLM → JSON Output
```

## ✨ Features

### 📁 Input Support
- **Image formats**: JPEG, PNG, TIFF, BMP
- **PDF support**: Single and multi-page PDFs with automatic page extraction
- **Batch processing**: Multiple files at once
- **Format validation**: Automatic file validation with helpful error messages

### 🔧 Preprocessing Steps
All steps are configurable and can be enabled/disabled:

1. **Grayscale conversion** - Convert color images to grayscale
2. **Noise removal** - Gaussian, Median, or Bilateral filtering
3. **Contrast enhancement** - CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. **Thresholding** - Adaptive (handwritten) or OTSU (printed text)
5. **Deskewing** - Automatic rotation correction using Hough transforms
6. **DPI normalization** - Resize to 300 DPI standard for OCR
7. **Morphological operations** *(optional)* - Strengthen text appearance
8. **Shadow removal** *(experimental)* - Remove scanning shadows
9. **Border cropping** *(optional)* - Remove white borders

### 🎛️ Configuration Presets
- **high_quality_printed** - Optimized for crisp printed prescriptions
- **handwritten** - Optimized for handwritten prescriptions
- **low_quality** - For noisy or low-quality scans
- **minimal** - Light processing for already clean images

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Note: You may need to install poppler for PDF support
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
# Linux: sudo apt-get install poppler-utils
# macOS: brew install poppler
```

### Basic Usage

```python
from preprocessing import preprocess_image

# Process a single image with default settings
results = preprocess_image("prescription.jpg")

# Process with custom settings
results = preprocess_image(
    input_path="prescription.pdf",
    preset="handwritten",  # Use preset for handwritten text
    save_outputs=True,     # Save processed images to disk
    debug=True            # Save intermediate processing steps
)

# Each result contains:
for result in results:
    processed_image = result['image']           # numpy array
    processing_info = result['info']            # processing details
    output_path = result['output_path']         # saved file path
    page_number = result['page_number']         # page number (for PDFs)
```

### Advanced Configuration

```python
from preprocessing import preprocess_image, PreprocessingConfig

# Create custom configuration
config = PreprocessingConfig()
config.target_dpi = 300
config.enable_deskewing = True
config.thresholding_method = "adaptive"
config.noise_removal_method = "bilateral"

# Process with custom config
results = preprocess_image("prescription.png", config=config)
```

## 📖 Detailed Usage Guide

### Configuration System

The preprocessing pipeline is highly configurable through the `PreprocessingConfig` class:

```python
from preprocessing.config import PreprocessingConfig, PresetConfigs

# Use default configuration
config = PreprocessingConfig()

# Or use a preset
config = PresetConfigs.get_handwritten_optimized()

# Customize specific parameters
config.clahe_clip_limit = 3.0
config.adaptive_block_size = 15
config.enable_morphological = True
```

### Key Configuration Parameters

#### General Settings
- `target_dpi`: Target DPI for OCR (default: 300)
- `debug_mode`: Save intermediate processing steps
- `save_outputs`: Save final processed images

#### Step Controls
```python
# Enable/disable preprocessing steps
config.enable_grayscale = True
config.enable_noise_removal = True
config.enable_contrast_enhancement = True
config.enable_thresholding = True
config.enable_deskewing = True
config.enable_resize = True
config.enable_morphological = False    # Optional
config.enable_shadow_removal = False   # Experimental
config.enable_border_crop = False      # Optional
```

#### Noise Removal
```python
config.noise_removal_method = "bilateral"  # "gaussian", "median", "bilateral"
config.bilateral_sigma_color = 75.0
config.bilateral_sigma_space = 75.0
```

#### Thresholding
```python
config.thresholding_method = "adaptive"  # "adaptive", "otsu", "auto"
config.adaptive_block_size = 11
config.adaptive_c_constant = 2.0
```

#### Contrast Enhancement (CLAHE)
```python
config.clahe_clip_limit = 3.0
config.clahe_tile_grid_size = (8, 8)
```

### Batch Processing

```python
from preprocessing import preprocess_batch

# Process multiple files
input_files = ["doc1.pdf", "doc2.jpg", "doc3.png"]
all_results = preprocess_batch(
    input_paths=input_files,
    preset="high_quality_printed",
    output_dir="./processed_images"
)

print(f"Processed {len(all_results)} total images")
```

### PDF Processing

PDFs are automatically detected and converted to images:

```python
# Process multi-page PDF
results = preprocess_image("prescription.pdf")

# Each page becomes a separate result
for result in results:
    page_num = result['page_number']
    image = result['image']
    print(f"Page {page_num}: {image.shape}")
```

### Output and Debug Information

```python
results = preprocess_image("test.jpg", debug=True)

result = results[0]
info = result['info']

print(f"Original shape: {info['original_shape']}")
print(f"Final shape: {info['final_shape']}")
print(f"Processing time: {info['processing_time_seconds']:.2f}s")
print(f"Steps applied: {', '.join(info['steps_applied'])}")

# Check for warnings
if info['warnings']:
    for warning in info['warnings']:
        print(f"Warning: {warning}")
```

## 🛠️ Development and Integration

### For Teammates Using This Module

1. **Install the module**: Copy the `preprocessing/` folder to your project
2. **Install dependencies**: `pip install -r preprocessing/requirements.txt`
3. **Import and use**:
   ```python
   from preprocessing import preprocess_image
   results = preprocess_image("your_prescription.pdf")
   ```

### Integration with OCR Systems

The module outputs OpenCV images (numpy arrays) that can be directly used with OCR libraries:

```python
# Preprocessing
from preprocessing import preprocess_image
results = preprocess_image("prescription.pdf")

# Your OCR integration
for result in results:
    processed_image = result['image']
    
    # Example: PaddleOCR
    # ocr_result = paddle_ocr.ocr(processed_image)
    
    # Example: TrOCR
    # ocr_result = trocr_model.predict(processed_image)
    
    # Example: Tesseract
    # text = pytesseract.image_to_string(processed_image)
```

### File Structure

```
preprocessing/
├── preprocess.py          # Main pipeline interface
├── pdf_utils.py           # PDF → image conversion
├── image_utils.py         # OpenCV preprocessing functions
├── config.py              # Configuration classes and presets
├── README.md              # This documentation
├── requirements.txt       # Python dependencies
└── examples/
    ├── input/             # Sample input files (add your own)
    └── output/            # Processed output files
```

## 🔧 Configuration Reference

### Predefined Presets

#### High Quality Printed (`"high_quality_printed"`)
Optimized for crisp, printed prescriptions:
- OTSU thresholding
- Gaussian noise removal
- Minimal morphological operations
- Lower CLAHE clip limit (2.0)

#### Handwritten (`"handwritten"`)
Optimized for handwritten prescriptions:
- Adaptive thresholding
- Bilateral noise removal
- Morphological operations enabled
- Higher CLAHE clip limit (4.0)
- Larger adaptive block size

#### Low Quality (`"low_quality"`)
For noisy or poor-quality scans:
- Strong bilateral filtering
- Enhanced CLAHE processing
- Morphological operations
- Aggressive noise removal

#### Minimal (`"minimal"`)
Light processing for clean images:
- Basic grayscale and thresholding only
- No noise removal or enhancement
- Fastest processing

### Custom Configuration Examples

#### For Dark or Low-Contrast Images
```python
config = PreprocessingConfig()
config.clahe_clip_limit = 5.0  # More aggressive enhancement
config.enable_contrast_enhancement = True
config.enable_morphological = True
```

#### For High-Resolution Images
```python
config = PreprocessingConfig()
config.target_dpi = 600  # Higher resolution
config.gaussian_kernel_size = (5, 5)  # Larger blur kernel
config.adaptive_block_size = 21  # Larger thresholding blocks
```

#### For Batch Processing Speed
```python
config = PreprocessingConfig()
config.enable_deskewing = False  # Skip slow deskewing
config.enable_shadow_removal = False  # Skip experimental steps
config.resize_method = "scale_factor"  # Faster than DPI calculation
config.scale_factor = 0.5  # Reduce size for speed
```

## 📊 Performance Considerations

### Processing Speed
- **Typical image (2000x3000)**: ~2-5 seconds
- **PDF (10 pages)**: ~20-40 seconds
- **Batch processing**: Linear scaling

### Memory Usage
- **Peak memory**: ~3x input file size
- **PDF processing**: Processes pages sequentially to limit memory

### Speed Optimization Tips
1. **Disable unnecessary steps** in your configuration
2. **Use lower DPI** for faster processing (e.g., 200 instead of 300)
3. **Skip deskewing** if images are already straight
4. **Use "minimal" preset** for pre-cleaned images

## 🚨 Troubleshooting

### Common Issues

#### "pdf2image not available" Error
```bash
# Install PDF support
pip install pdf2image

# Install system dependencies:
# Windows: Download poppler from GitHub
# Linux: sudo apt-get install poppler-utils  
# macOS: brew install poppler
```

#### OpenCV Installation Issues
```bash
# Try different OpenCV packages
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### Memory Issues with Large PDFs
```python
# Process pages individually for large PDFs
from preprocessing.pdf_utils import PDFProcessor

processor = PDFProcessor(dpi=200)  # Lower DPI
images = processor.convert_pdf_to_images("large.pdf")
```

#### Poor Results on Handwritten Text
```python
# Try handwritten preset with custom parameters
config = PresetConfigs.get_handwritten_optimized()
config.adaptive_block_size = 21  # Larger block size
config.adaptive_c_constant = 5.0  # Higher constant
config.clahe_clip_limit = 6.0  # More enhancement
```

### Debugging

Enable debug mode to see intermediate steps:

```python
results = preprocess_image("problem.jpg", debug=True)
# Check the debug/ folder in output directory for intermediate images
```

Check processing info for warnings:
```python
result = results[0]
if result['info']['warnings']:
    print("Warnings:", result['info']['warnings'])
```

## 🤝 Contributing

### Adding New Preprocessing Steps

1. Add the step to `image_utils.py`:
```python
def your_new_step(self, image: np.ndarray) -> np.ndarray:
    """Your new preprocessing step."""
    # Implement your logic here
    return processed_image
```

2. Add configuration parameters to `config.py`:
```python
# In PreprocessingConfig class
enable_your_step: bool = False
your_step_parameter: float = 1.0
```

3. Integrate in the pipeline (`image_utils.py` `process_image` method):
```python
# Step N: Your new step
if self.config.enable_your_step:
    current_image = self.your_new_step(current_image)
    processing_info['steps_applied'].append('your_step')
```

### Testing

Add test images to `examples/input/` and run:
```python
from preprocessing import preprocess_image

# Test your changes
results = preprocess_image("examples/input/test.jpg", debug=True)
```

## 📄 License and Attribution

This preprocessing module was created for the Eyeglass Prescription Image Reader project. 

**Author**: Computer Vision Engineer  
**Date**: January 21, 2026  
**Version**: 1.0.0

## 🔗 Dependencies Attribution

- **OpenCV**: BSD License - Computer vision and image processing
- **NumPy**: BSD License - Numerical computing
- **Pillow**: HPND License - Image handling
- **pdf2image**: MIT License - PDF to image conversion
- **PyMuPDF**: AGPL License - Alternative PDF processing

---

For questions, issues, or suggestions, please contact the development team or create an issue in the project repository.