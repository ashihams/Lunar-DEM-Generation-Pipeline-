# NASA Lunar Pipeline

A comprehensive, modular pipeline for processing lunar imagery data with high-performance image preprocessing, correction, and enhancement capabilities.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated classes for each functionality
- **High-Performance Processing**: GPU-accelerated super resolution and parallel processing
- **Multiple Image Formats**: Support for PNG, JPG, TIFF, and IMG formats common in planetary data
- **Radiometric Correction**: Dark current and flat field correction for raw data
- **Geometric Correction**: Lens distortion correction with camera calibration
- **Super Resolution**: AI-powered image enhancement for Wide Angle Camera (WAC) images
- **Image Stitching**: Feature-based stitching for Narrow Angle Camera (NAC) images
- **Robust Error Handling**: Comprehensive logging and graceful failure handling

## 📁 Project Structure

```
nasa-lunar-pipeline/
├── src/                          # Source code package
│   ├── __init__.py              # Package initialization
│   ├── enums.py                 # Enumerations (DataProductType, CameraType)
│   ├── models.py                # Data models (ImageMetadata)
│   ├── preprocessor.py          # Image preprocessing (LunarImagePreprocessor)
│   ├── corrector.py             # Image correction (RadiometricCorrector, GeometricCorrector)
│   ├── processor.py             # Image processing (SuperResolutionProcessor, ImageStitcher)
│   ├── parallel.py              # Parallel processing (ParallelProcessor)
│   └── pipeline.py              # Main pipeline orchestrator (LunarDLTPipeline)
├── main.py                      # Main entry point
├── config.json                  # Configuration file (auto-generated)
├── requirements.txt             # Python dependencies
├── setup_lunar.sql             # Database setup
├── docker-compose.yml          # Docker configuration
├── Dockerfile                  # Docker image definition
└── README.md                   # This file
```

## 🏗️ Architecture

### Core Classes

1. **`DataProductType` & `CameraType`** (`src/enums.py`)
   - Enumerations for data product types (RAW, CALIBRATION, DERIVED)
   - Camera system types (WAC - Wide Angle, NAC - Narrow Angle)

2. **`ImageMetadata`** (`src/models.py`)
   - Dataclass for storing image metadata and properties
   - Includes product ID, camera type, resolution, timestamps, etc.

3. **`LunarImagePreprocessor`** (`src/preprocessor.py`)
   - Manages super resolution models and camera calibration
   - Handles device selection (CPU/GPU) and model initialization

4. **`RadiometricCorrector`** (`src/corrector.py`)
   - Applies dark current correction
   - Implements flat field correction
   - Handles complete radiometric correction pipeline

5. **`GeometricCorrector`** (`src/corrector.py`)
   - Lens distortion correction using camera calibration parameters
   - Handles large image resizing for OpenCV compatibility
   - Robust error handling for calibration failures

6. **`SuperResolutionProcessor`** (`src/processor.py`)
   - GPU-accelerated super resolution using SRCNN model
   - Optimized for WAC images requiring enhancement
   - Tensor normalization and GPU memory management

7. **`ImageStitcher`** (`src/processor.py`)
   - Feature-based image stitching using SIFT and RANSAC
   - Designed for NAC image mosaicking
   - Homography calculation and perspective warping

8. **`ParallelProcessor`** (`src/parallel.py`)
   - Thread-based parallel processing manager
   - Configurable worker count and batch processing
   - Memory-efficient batch handling

9. **`LunarDLTPipeline`** (`src/pipeline.py`)
   - Main orchestrator class
   - Coordinates all processing components
   - Handles file loading, metadata parsing, and output management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for super resolution)
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd nasa-lunar-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

### Configuration

The pipeline automatically creates a `config.json` file with default settings:

```json
{
  "batch_size": 8,
  "max_workers": null,
  "super_resolution": {
    "scale_factor": 2,
    "model_path": null
  },
  "output_format": "png",
  "quality_metrics": true,
  "gpu_memory_limit": 0.8
}
```

## 📖 Usage Examples

### Basic Usage

```python
from src.pipeline import LunarDLTPipeline

# Initialize pipeline
pipeline = LunarDLTPipeline('config.json')

# Process a single image
image, metadata = pipeline.load_image('path/to/lunar_image.png')
processed_image = pipeline.preprocess_single_image(image, metadata)

# Process entire dataset
results = pipeline.process_dataset('./input_images', './output_images')
print(f"Processed {results['processed']} images")
```

### Custom Processing

```python
from src.corrector import RadiometricCorrector, GeometricCorrector
from src.processor import SuperResolutionProcessor

# Use individual components
radiometric_corrector = RadiometricCorrector()
geometric_corrector = GeometricCorrector(calibration_params)
super_res_processor = SuperResolutionProcessor(model, device)

# Apply corrections
corrected_image = radiometric_corrector.apply_radiometric_correction(image)
undistorted_image = geometric_corrector.undistort_image(corrected_image, 'wac')
enhanced_image = super_res_processor.enhance_resolution(undistorted_image)
```

## 🔧 Customization

### Adding New Image Formats

Extend the `_load_img_format` method in `LunarDLTPipeline`:

```python
def _load_custom_format(self, file_path: str) -> np.ndarray:
    # Your custom loading logic here
    pass
```

### Custom Super Resolution Models

Modify the `SRCNN` class in `src/preprocessor.py` or create new model classes.

### Additional Corrections

Extend the correction classes in `src/corrector.py` with new correction methods.

## 📊 Performance

- **GPU Acceleration**: Super resolution processing on CUDA devices
- **Parallel Processing**: Multi-threaded batch processing
- **Memory Optimization**: Efficient batch handling and image resizing
- **Error Recovery**: Graceful handling of corrupted or unsupported files

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config.json
2. **Large Image Errors**: Images are automatically resized for processing
3. **File Loading Failures**: Check file paths and supported formats

### Logging

The pipeline provides comprehensive logging. Check console output for detailed processing information and error messages.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NASA for lunar imagery data
- OpenCV and PyTorch communities
- Planetary Data System (PDS) standards

## 📞 Support

For questions or issues, please open an issue on the GitHub repository or contact the development team. 