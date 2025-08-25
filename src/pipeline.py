"""
Main Pipeline for NASA Lunar Pipeline

This module contains the main LunarDLTPipeline class which orchestrates
the complete lunar imagery processing pipeline.
"""

import os
import numpy as np
import cv2
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image

from .enums import DataProductType, CameraType
from .models import ImageMetadata
from .preprocessor import LunarImagePreprocessor
from .corrector import RadiometricCorrector, GeometricCorrector
from .processor import SuperResolutionProcessor, ImageStitcher
from .parallel import ParallelProcessor

logger = logging.getLogger(__name__)


class LunarDLTPipeline:
    """Main DLT pipeline orchestrator"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize components
        self.preprocessor = LunarImagePreprocessor(self.config)
        self.radiometric_corrector = RadiometricCorrector()
        self.geometric_corrector = GeometricCorrector(
            self.preprocessor.camera_calibration_params
        )
        self.super_resolution_processor = SuperResolutionProcessor(
            self.preprocessor.super_resolution_model,
            self.preprocessor.device
        )
        self.image_stitcher = ImageStitcher()
        self.parallel_processor = ParallelProcessor()

        logger.info("DLT Pipeline initialized successfully")

    def parse_product_id(self, product_path: str) -> ImageMetadata:
        """Parse product ID to extract metadata"""
        # Example: ch2_ohr_ncp_20250304T0456267027_d_img_d18
        filename = Path(product_path).name

        # Parse components (this is simplified - adjust based on actual format)
        parts = filename.split('_')

        # Determine camera type
        camera_type = CameraType.WAC if 'wac' in filename.lower() else CameraType.NAC

        # Determine product type based on naming convention
        if '_raw_' in filename or 'd_img' in filename:
            product_type = DataProductType.RAW
        elif '_cal_' in filename or 'calibrated' in filename:
            product_type = DataProductType.CALIBRATION
        else:
            product_type = DataProductType.DERIVED

        # Extract timestamp
        timestamp = None
        for part in parts:
            if 'T' in part and len(part) > 10:
                timestamp = part
                break

        return ImageMetadata(
            product_id=filename,
            camera_type=camera_type,
            product_type=product_type,
            timestamp=timestamp or "unknown",
            resolution=(0, 0),  # Will be updated when image is loaded
            file_path=product_path
        )

    def load_image(self, file_path: str) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image and extract metadata with robust error handling"""
        logger.info(f"Loading image: {file_path}")

        image = None
        actual_path = file_path

        # First, check if file exists as-is
        if not os.path.exists(file_path):
            # Try adding common extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.img']:
                test_path = file_path + ext
                if os.path.exists(test_path):
                    actual_path = test_path
                    break
            else:
                logger.error(f"File not found: {file_path}")
                return None, None

        # Get file info
        file_size = os.path.getsize(actual_path)
        logger.info(f"File size: {file_size} bytes")

        # Try loading based on file extension
        file_ext = Path(actual_path).suffix.lower()

        try:
            if file_ext == '.img' or 'img' in file_ext:
                logger.info("Loading as IMG format")
                image = self._load_img_format(actual_path)
            else:
                logger.info(f"Loading as standard image format: {file_ext}")
                # Try OpenCV first
                image = cv2.imread(actual_path, cv2.IMREAD_UNCHANGED)

                if image is None:
                    # Try PIL as fallback
                    with Image.open(actual_path) as pil_image:
                        image = np.array(pil_image)
                        # Convert RGB to BGR for OpenCV compatibility
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"Failed to load with specific method: {e}")
            # Try as raw binary data
            try:
                logger.info("Attempting to load as raw binary")
                image = self._load_raw_binary(actual_path)
            except Exception as e2:
                logger.error(f"All loading methods failed: {e2}")
                return None, None

        if image is None:
            logger.error(f"Could not load image: {actual_path}")
            return None, None

        # Log image properties
        logger.info(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")

        # Ensure image is in a processable format
        if len(image.shape) == 2:
            # Grayscale
            pass
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel, squeeze
                image = image.squeeze(axis=2)
            elif image.shape[2] == 3:
                # RGB/BGR, keep as is
                pass
            elif image.shape[2] == 4:
                # RGBA, convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            logger.error(f"Unsupported image shape: {image.shape}")
            return None, None

        # Create metadata
        metadata = self.parse_product_id(actual_path)
        metadata.resolution = (image.shape[1], image.shape[0])

        logger.info(f"Successfully loaded: {metadata.product_id} - {metadata.camera_type.value} - {metadata.product_type.value}")

        return image, metadata

    def _load_img_format(self, file_path: str) -> np.ndarray:
        """Load IMG format common in planetary data"""
        try:
            # First, try to read the PDS label if it exists
            label_info = self._parse_pds_label(file_path)

            with open(file_path, 'rb') as f:
                # Use label info if available, otherwise estimate
                if label_info:
                    f.seek(label_info.get('record_bytes', 0) * label_info.get('label_records', 0))
                    width = label_info.get('line_samples', 1024)
                    height = label_info.get('lines', 1024)
                    sample_bits = label_info.get('sample_bits', 8)

                    if sample_bits == 8:
                        dtype = np.uint8
                    elif sample_bits == 16:
                        dtype = np.uint16
                    else:
                        dtype = np.uint8

                    data = np.fromfile(f, dtype=dtype, count=width*height)

                    if len(data) == width * height:
                        image = data.reshape((height, width))
                    else:
                        raise ValueError(f"Data size mismatch: expected {width*height}, got {len(data)}")

                else:
                    # Fallback to estimation method
                    file_size = os.path.getsize(file_path)

                    # Try different header sizes and data types
                    for header_size in [0, 512, 1024, 2048, 4096]:
                        remaining_bytes = file_size - header_size

                        for dtype, byte_size in [(np.uint8, 1), (np.uint16, 2), (np.float32, 4)]:
                            num_pixels = remaining_bytes // byte_size

                            # Try common square sizes first
                            size = int(np.sqrt(num_pixels))
                            if size * size == num_pixels and size >= 64:  # Reasonable minimum size
                                f.seek(header_size)
                                data = np.fromfile(f, dtype=dtype, count=num_pixels)
                                image = data.reshape((size, size))
                                break

                            # Try common rectangular sizes
                            for ratio in [(4, 3), (3, 2), (16, 9), (2, 1)]:
                                h = int(np.sqrt(num_pixels * ratio[1] / ratio[0]))
                                w = int(num_pixels / h)
                                if h * w == num_pixels and h >= 64 and w >= 64:
                                    f.seek(header_size)
                                    data = np.fromfile(f, dtype=dtype, count=num_pixels)
                                    image = data.reshape((h, w))
                                    break
                            else:
                                continue
                            break
                        else:
                            continue
                        break
                    else:
                        raise ValueError("Could not determine image dimensions and format")

                # Normalize to 8-bit if needed
                if image.dtype != np.uint8:
                    image_min, image_max = image.min(), image.max()
                    if image_max > image_min:
                        image = ((image.astype(np.float32) - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                    else:
                        image = np.zeros_like(image, dtype=np.uint8)

                # Validate reasonable image dimensions
                h, w = image.shape
                if h > 32767 or w > 32767:
                    logger.warning(f"Image too large ({h}x{w}), resizing for processing")
                    # Resize to manageable size while maintaining aspect ratio
                    max_dim = 8192
                    scale = min(max_dim / h, max_dim / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                return image

        except Exception as e:
            logger.error(f"Error loading IMG format: {e}")
            raise

    def _parse_pds_label(self, file_path: str) -> Dict:
        """Parse PDS (Planetary Data System) label information"""
        label_info = {}
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(8192)  # Read first 8KB for label

                # Look for common PDS keywords
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if '=' in line and not line.startswith('/*'):
                        try:
                            key, value = line.split('=', 1)
                            key = key.strip().lower().replace(' ', '_')
                            value = value.strip().rstrip(';').strip('"').strip("'")

                            # Convert numeric values
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except ValueError:
                                pass  # Keep as string

                            label_info[key] = value
                        except ValueError:
                            continue

        except Exception as e:
            logger.debug(f"Could not parse PDS label: {e}")

        return label_info

    def _load_raw_binary(self, file_path: str) -> np.ndarray:
        """Load raw binary image data"""
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint8)

        # Try to determine dimensions
        size = int(np.sqrt(len(data)))
        if size * size == len(data):
            return data.reshape((size, size))
        else:
            # Try common aspect ratios
            for ratio in [(4, 3), (16, 9), (3, 2), (1, 1)]:
                h = int(np.sqrt(len(data) * ratio[1] / ratio[0]))
                w = int(len(data) / h)
                if h * w == len(data):
                    return data.reshape((h, w))

        raise ValueError("Could not determine image dimensions from raw data")

    def preprocess_single_image(self, image: np.ndarray, metadata: ImageMetadata) -> np.ndarray:
        """Preprocess a single image based on its metadata"""
        logger.info(f"Processing {metadata.product_id} - {metadata.camera_type.value} - {metadata.product_type.value}")

        processed_image = image.copy()

        # Apply corrections based on product type
        if metadata.product_type == DataProductType.RAW:
            # Apply both radiometric and geometric corrections
            processed_image = self.radiometric_corrector.apply_radiometric_correction(processed_image)
            processed_image = self.geometric_corrector.undistort_image(
                processed_image, metadata.camera_type.value
            )
        elif metadata.product_type == DataProductType.CALIBRATION:
            # Apply only geometric correction
            processed_image = self.geometric_corrector.undistort_image(
                processed_image, metadata.camera_type.value
            )
        # DERIVED data needs no correction

        # Apply camera-specific enhancements
        if metadata.camera_type == CameraType.WAC:
            # Apply super resolution for WAC images (lower clarity)
            processed_image = self.super_resolution_processor.enhance_resolution(processed_image)

        return processed_image

    def process_dataset(self, dataset_path: str, output_path: str) -> Dict:
        """Process entire dataset with parallel processing"""
        logger.info(f"Starting dataset processing: {dataset_path}")

        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Collect all image files
        image_files = []
        dataset_path = Path(dataset_path)

        if dataset_path.is_file():
            image_files = [str(dataset_path)]
        else:
            # Scan directory for image files
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.img']:
                image_files.extend(dataset_path.glob(ext))
            image_files = [str(f) for f in image_files]

        if not image_files:
            logger.warning(f"No image files found in {dataset_path}")
            return {"status": "no_files", "processed": 0}

        # Load and process images
        processed_count = 0
        processing_stats = {
            "total_files": len(image_files),
            "processed": 0,
            "failed": 0,
            "processing_time": 0,
            "camera_types": {},
            "product_types": {}
        }

        start_time = time.time()

        # Process in batches for memory efficiency
        batch_size = self.config.get('batch_size', 8)

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]

            # Load batch
            batch_data = []
            for file_path in batch_files:
                try:
                    image, metadata = self.load_image(file_path)
                    if image is not None:
                        batch_data.append((image, metadata))
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    processing_stats["failed"] += 1

            if not batch_data:
                continue

            # Process batch in parallel
            try:
                processed_images = self.parallel_processor.process_batch_parallel(
                    batch_data, self._process_wrapper
                )

                # Save processed images
                for j, ((original_image, metadata), processed_image) in enumerate(zip(batch_data, processed_images)):
                    if processed_image is not None:
                        output_file = Path(output_path) / f"processed_{metadata.product_id}.png"
                        cv2.imwrite(str(output_file), processed_image)

                        processed_count += 1
                        processing_stats["processed"] += 1

                        # Update statistics
                        cam_type = metadata.camera_type.value
                        prod_type = metadata.product_type.value
                        processing_stats["camera_types"][cam_type] = processing_stats["camera_types"].get(cam_type, 0) + 1
                        processing_stats["product_types"][prod_type] = processing_stats["product_types"].get(prod_type, 0) + 1

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                processing_stats["failed"] += len(batch_data)

        processing_stats["processing_time"] = time.time() - start_time

        logger.info(f"Processing complete. Processed: {processed_count}, Failed: {processing_stats['failed']}")
        logger.info(f"Total processing time: {processing_stats['processing_time']:.2f} seconds")

        return processing_stats

    def _process_wrapper(self, image: np.ndarray, metadata: ImageMetadata) -> np.ndarray:
        """Wrapper for single image processing to handle exceptions"""
        try:
            logger.info(f"Processing wrapper called for: {metadata.product_id}")
            result = self.preprocess_single_image(image, metadata)
            if result is not None:
                logger.info(f"Successfully processed: {metadata.product_id}")
            else:
                logger.warning(f"Processing returned None for: {metadata.product_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to process {metadata.product_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None


def create_default_config():
    """Create default configuration file"""
    config = {
        "batch_size": 8,
        "max_workers": None,
        "super_resolution": {
            "scale_factor": 2,
            "model_path": None
        },
        "output_format": "png",
        "quality_metrics": True,
        "gpu_memory_limit": 0.8
    }

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return config 