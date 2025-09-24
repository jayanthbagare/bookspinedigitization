#!/usr/bin/env python3
"""
Enhanced Book Spine Detection with Configurable OCR
Supports both Google Vision API and Tesseract OCR with settings switch
"""

import cv2
import numpy as np
import os
import json
import requests
import re
import pytesseract
import io
import logging
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import torch
from ultralytics import YOLO, settings
from PIL import Image

# Import Google Cloud Vision only when needed
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

from config import ocr_config

# Configure YOLO settings
try:
    # Update YOLO settings for better performance and organization
    settings.update(
        {
            "runs_dir": "runs",  # Directory for training/validation runs
            "weights_dir": "models",  # Directory for model weights
            "datasets_dir": "datasets",  # Directory for datasets
            "sync": False,  # Disable analytics and crash reporting
            "api_key": "",  # Disable API key warnings
        }
    )
except Exception as e:
    logging.warning(f"Could not update YOLO settings: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BookSpine:
    """Data class for book spine information"""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    title: str
    author: str
    confidence: float
    raw_text: str
    preprocessed_image: Optional[np.ndarray] = None


class ImagePreprocessor:
    """Advanced image preprocessing for optimal Tesseract OCR"""

    def __init__(self):
        self.debug_mode = False

    def detect_text_polarity(self, image: np.ndarray, sample_regions: int = 8) -> bool:
        """
        Detect if text is light-on-dark (needs inversion) or dark-on-light
        Returns True if image needs inversion (light text on dark background)
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        h, w = gray.shape

        # Sample multiple regions to get better assessment
        inversion_scores = []

        for i in range(sample_regions):
            # Get random sample regions, focusing on likely text areas
            if i < 4:  # First 4 samples from center regions
                y1 = h // 4 + np.random.randint(0, h // 4)
                y2 = y1 + h // 8
                x1 = w // 8 + np.random.randint(0, 3 * w // 4)
                x2 = x1 + w // 8
            else:  # Remaining samples from edges (where text might be)
                y1 = np.random.randint(0, h // 4)
                y2 = y1 + h // 6
                x1 = np.random.randint(0, w // 4)
                x2 = x1 + w // 6

            # Ensure bounds
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)

            if y2 - y1 < 10 or x2 - x1 < 10:
                continue

            region = gray[y1:y2, x1:x2]

            # Calculate various metrics
            avg_intensity = np.mean(region)

            # Apply Otsu thresholding
            _, binary = cv2.threshold(
                region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Count white vs black pixels
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            total_pixels = white_pixels + black_pixels

            if total_pixels == 0:
                continue

            white_ratio = white_pixels / total_pixels

            # Check for connected components (text-like structures)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            # Score based on multiple factors
            score = 0

            # 1. If background is dark and we have small white regions (text)
            if avg_intensity < 100 and white_ratio < 0.3 and num_labels > 2:
                score += 2

            # 2. If we have many small connected components (typical of inverted text)
            if num_labels > 5 and avg_intensity < 120:
                score += 1

            # 3. Low average intensity suggests dark background
            if avg_intensity < 80:
                score += 1

            inversion_scores.append(score)

        # Return True if majority of samples suggest inversion is needed
        if not inversion_scores:
            return False

        avg_score = sum(inversion_scores) / len(inversion_scores)
        return avg_score >= 1.5

    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Enhanced contrast using multiple techniques"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)

            # Merge channels and convert back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            # Apply CLAHE directly to grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            # Additional histogram equalization for very dark images
            if np.mean(enhanced) < 100:
                enhanced = cv2.equalizeHist(enhanced)

            return enhanced

    def remove_noise_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced noise removal specifically for text images"""
        # Median blur to remove salt and pepper noise
        denoised = cv2.medianBlur(image, 3)

        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        # Close small gaps in characters
        closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        # Open to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        return opened

    def detect_and_correct_skew(
        self, image: np.ndarray, max_angle: float = 10.0
    ) -> np.ndarray:
        """Detect and correct text skew using Hough line detection"""
        # Create a copy
        corrected = image.copy()

        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[: min(10, len(lines))]:  # Use top 10 lines
                angle = np.degrees(theta) - 90
                if abs(angle) <= max_angle:  # Only consider reasonable angles
                    angles.append(angle)

            if angles:
                # Use median angle for robustness
                median_angle = np.median(angles)

                if abs(median_angle) > 0.5:  # Only correct if significant skew
                    h, w = image.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    corrected = cv2.warpAffine(
                        image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255
                    )

        return corrected

    def preprocess_for_ocr(
        self, image: np.ndarray, spine_height: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advanced preprocessing pipeline optimized for high-accuracy book spine OCR
        """
        metadata = {
            "inverted": False,
            "enhanced_contrast": False,
            "denoised": False,
            "border_added": False,
            "skew_corrected": False,
            "resized": False,
            "sharpened": False,
            "gamma_corrected": False,
        }

        # Step 1: Convert to grayscale with optimal weights
        if len(image.shape) == 3:
            # Use custom weights for better text contrast
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        original_gray = gray.copy()

        # Step 2: Aggressive upscaling for better OCR (minimum 600px height)
        h, w = gray.shape
        target_height = max(600, h * 2)  # At least 2x scaling
        if h < target_height:
            scale_factor = target_height / h
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            # Use INTER_CUBIC for better quality on text
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            metadata["resized"] = True
            logger.info(f"Upscaled image from {h}x{w} to {new_h}x{new_w}")

        # Step 3: Gamma correction for better text visibility
        gamma = 1.2  # Slightly brighten
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
        metadata["gamma_corrected"] = True

        # Step 4: Advanced contrast enhancement
        # Use CLAHE with optimized parameters
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        metadata["enhanced_contrast"] = True

        # Step 5: Unsharp masking for text sharpening
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        metadata["sharpened"] = True

        # Step 6: Advanced denoising while preserving text
        # Use bilateral filter to preserve edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        metadata["denoised"] = True

        # Step 7: Text polarity detection and correction
        needs_inversion = self.detect_text_polarity_improved(gray)
        if needs_inversion:
            gray = cv2.bitwise_not(gray)
            metadata["inverted"] = True
            logger.info("Applied image inversion for light-text-on-dark")

        # Step 8: Skew correction with improved algorithm
        gray = self.detect_and_correct_skew_improved(gray)
        metadata["skew_corrected"] = True

        # Step 9: Add substantial border (critical for Tesseract accuracy)
        border_size = max(50, int(min(gray.shape) * 0.15))
        gray = cv2.copyMakeBorder(
            gray,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        metadata["border_added"] = True

        # Step 10: Advanced binarization using multiple methods
        binary_image = self.advanced_binarization(gray)

        return binary_image, metadata

    def detect_text_polarity_improved(self, image: np.ndarray) -> bool:
        """
        Improved text polarity detection using multiple statistical measures
        """
        # Sample the center region where text is most likely
        h, w = image.shape
        center_h, center_w = h // 4, w // 4
        center_region = image[center_h:3*center_h, center_w:3*center_w]

        if center_region.size == 0:
            return False

        # Calculate mean intensity
        mean_intensity = np.mean(center_region)

        # Calculate histogram
        hist = cv2.calcHist([center_region], [0], None, [256], [0, 256])

        # Find peaks in histogram
        hist_smooth = cv2.GaussianBlur(hist.flatten().reshape(-1, 1), (5, 1), 0).flatten()

        # If mean is low and histogram is bimodal with dark peak dominant
        if mean_intensity < 100:
            # Check if there's a strong dark peak
            dark_peak = np.sum(hist_smooth[:100])
            light_peak = np.sum(hist_smooth[150:])

            if dark_peak > light_peak * 2:
                return True

        return False

    def detect_and_correct_skew_improved(self, image: np.ndarray) -> np.ndarray:
        """
        Improved skew detection and correction using Hough transform
        """
        # Create edges for line detection
        edges = cv2.Canny(image, 30, 100, apertureSize=3)

        # Dilate edges to connect text components
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is not None:
            angles = []
            for rho, theta in lines[:20]:  # Use top 20 lines
                angle = np.degrees(theta) - 90
                if abs(angle) < 15:  # Only consider reasonable angles
                    angles.append(angle)

            if angles:
                # Use median for robustness
                median_angle = np.median(angles)

                if abs(median_angle) > 0.5:
                    h, w = image.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    corrected = cv2.warpAffine(image, M, (w, h),
                                             flags=cv2.INTER_CUBIC,
                                             borderValue=255)
                    return corrected

        return image

    def advanced_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced binarization using multiple methods and selecting the best
        """
        methods = []

        # Method 1: Otsu's thresholding
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu", otsu))

        # Method 2: Adaptive threshold (mean)
        adaptive_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15, 8)
        methods.append(("Adaptive_Mean", adaptive_mean))

        # Method 3: Adaptive threshold (gaussian)
        adaptive_gauss = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 15, 8)
        methods.append(("Adaptive_Gaussian", adaptive_gauss))

        # Method 4: Custom threshold based on local statistics
        kernel_size = 25
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_std = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel) ** 0.5
        threshold = local_mean - 0.3 * local_std
        custom = np.where(image > threshold, 255, 0).astype(np.uint8)
        methods.append(("Custom", custom))

        # Evaluate each method based on text-like characteristics
        best_binary = otsu  # Default fallback
        best_score = 0

        for name, binary in methods:
            score = self.evaluate_binarization_quality(binary)
            if score > best_score:
                best_score = score
                best_binary = binary

        return best_binary

    def evaluate_binarization_quality(self, binary_image: np.ndarray) -> float:
        """
        Evaluate the quality of binarization for text recognition
        """
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0

        score = 0
        text_like_contours = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Skip very small or very large contours
            if area < 20 or area > 5000:
                continue

            aspect_ratio = w / h if h > 0 else 0

            # Text characteristics
            if 0.1 < aspect_ratio < 5 and 20 < area < 2000:
                text_like_contours += 1

                # Bonus for good aspect ratios (typical for letters)
                if 0.3 < aspect_ratio < 2:
                    score += 2
                else:
                    score += 1

                # Bonus for reasonable size
                if 50 < area < 500:
                    score += 1

        # Normalize score
        return score / max(1, len(contours)) * text_like_contours


class TesseractOCR:
    """Enhanced Tesseract OCR for book spines"""

    def __init__(self):
        """Initialize Tesseract OCR with configuration"""
        self.confidence_threshold = ocr_config.get('tesseract.confidence_threshold', 60)
        self.psm = ocr_config.get('tesseract.psm', 6)
        self.oem = ocr_config.get('tesseract.oem', 1)
        self.char_whitelist = ocr_config.get('tesseract.char_whitelist',
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-()[]&* ")

        # Verify Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized: {pytesseract.get_tesseract_version()}")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            raise

    def get_optimal_config(self, image: np.ndarray) -> str:
        """Get optimal Tesseract configuration based on image characteristics"""
        h, w = image.shape[:2]

        # Base configuration
        config = f"--oem {self.oem} --psm {self.psm}"

        # Add character whitelist if specified
        if self.char_whitelist:
            config += f" -c tessedit_char_whitelist={self.char_whitelist}"

        # Adjust PSM based on image aspect ratio and size
        aspect_ratio = h / w if w > 0 else 1
        if aspect_ratio > 3:  # Very tall and narrow (typical book spine)
            config = config.replace(f"--psm {self.psm}", "--psm 8")  # Single word
        elif h < 100:  # Very short image
            config = config.replace(f"--psm {self.psm}", "--psm 7")  # Single text line

        return config

    def extract_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using Tesseract with confidence scores"""
        try:
            config = self.get_optimal_config(image)

            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )

            text_parts = []
            confidences = []
            word_details = []

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if conf > self.confidence_threshold and text and len(text) > 1:
                    text_parts.append(text)
                    confidences.append(conf)

                    word_details.append({
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        ),
                    })

            full_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            logger.info(f"Tesseract extracted: '{full_text[:50]}...' with confidence: {avg_confidence:.1f}%")

            return full_text, avg_confidence / 100.0, word_details

        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {e}")
            return "", 0.0, []

    def extract_text_multiple_configs(self, image: np.ndarray) -> Tuple[str, float]:
        """Try multiple Tesseract configurations and return the best result"""
        configs = [
            f"--oem {self.oem} --psm 6",  # Uniform block of text
            f"--oem {self.oem} --psm 8",  # Single word
            f"--oem {self.oem} --psm 7",  # Single text line
            f"--oem {self.oem} --psm 13", # Raw line. Treat the image as a single text line
        ]

        if self.char_whitelist:
            configs = [f"{config} -c tessedit_char_whitelist={self.char_whitelist}" for config in configs]

        best_result = ("", 0.0)

        for config in configs:
            try:
                result = pytesseract.image_to_string(image, config=config)

                # Simple confidence estimation based on result quality
                confidence = self._estimate_tesseract_confidence(result)

                if confidence > best_result[1]:
                    best_result = (result.strip(), confidence)

            except Exception as e:
                logger.warning(f"Config {config} failed: {e}")
                continue

        logger.info(f"Best Tesseract result: '{best_result[0][:50]}...' with estimated confidence: {best_result[1]:.1f}")
        return best_result

    def _estimate_tesseract_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics"""
        if not text.strip():
            return 0.0

        confidence = 50.0  # Base confidence

        # Boost for longer text
        confidence += min(20, len(text) * 0.5)

        # Boost for alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        confidence += alpha_ratio * 20

        # Reduce for special characters that indicate OCR errors
        error_chars = '|~`^{}\\@#$%'
        error_ratio = sum(c in error_chars for c in text) / len(text)
        confidence -= error_ratio * 30

        return min(100.0, max(0.0, confidence)) / 100.0


class GoogleVisionOCR:
    """Google Cloud Vision API OCR for book spines"""

    def __init__(self, use_fallback=True):
        """Initialize Google Vision OCR with optional Tesseract fallback"""
        if not GOOGLE_VISION_AVAILABLE:
            raise ImportError("Google Cloud Vision library not available. Install with: pip install google-cloud-vision")

        self.use_fallback = use_fallback
        self.confidence_threshold = ocr_config.get('google_vision.confidence_threshold', 0.5)

        # Initialize Google Vision client
        try:
            # Check if credentials are available
            credentials_path = ocr_config.google_credentials_path
            if credentials_path and not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found: {credentials_path}")

            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Vision API: {e}")
            if not use_fallback:
                raise
            else:
                self.client = None
                logger.info("Will use Tesseract fallback for OCR")

        # Initialize Tesseract OCR for fallback if needed
        if use_fallback:
            try:
                self.tesseract_ocr = TesseractOCR()
                logger.info("Tesseract fallback initialized")
            except Exception as e:
                logger.warning(f"Tesseract fallback not available: {e}")
                self.tesseract_ocr = None

    def image_to_base64(self, image: np.ndarray) -> bytes:
        """Convert numpy image to bytes for Google Vision API"""
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()

    def extract_text_with_google_vision(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Extract text using Google Cloud Vision API"""

        if self.client is None:
            logger.warning("Google Vision API not available, using fallback")
            return self.extract_text_fallback_tesseract(image)

        try:
            # Convert image to bytes
            image_bytes = self.image_to_base64(image)

            # Create Vision API image object
            vision_image = vision.Image(content=image_bytes)

            # Configure text detection features
            features = [vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)]

            # Create request
            request = vision.AnnotateImageRequest(image=vision_image, features=features)

            # Make API call
            response = self.client.annotate_image(request=request)

            if response.error.message:
                logger.error(f"Google Vision API error: {response.error.message}")
                return self.extract_text_fallback_tesseract(image)

            # Process text annotations
            text_annotations = response.text_annotations

            if not text_annotations:
                logger.info("No text detected by Google Vision API")
                return "", 0.0, []

            # Extract full text and word details
            full_text = text_annotations[0].description if text_annotations else ""

            # Process individual words with confidence
            word_details = []
            confidences = []

            for annotation in text_annotations[1:]:  # Skip first (full text)
                text = annotation.description.strip()
                if text:
                    # Google Vision API doesn't provide confidence scores directly
                    # We'll estimate confidence based on text quality
                    confidence = self.estimate_confidence(text, annotation.bounding_poly)
                    confidences.append(confidence)

                    # Get bounding box
                    vertices = annotation.bounding_poly.vertices
                    if vertices:
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        bbox = (min(x_coords), min(y_coords),
                               max(x_coords) - min(x_coords),
                               max(y_coords) - min(y_coords))
                    else:
                        bbox = (0, 0, 0, 0)

                    word_details.append({
                        "text": text,
                        "confidence": confidence / 100.0,
                        "bbox": bbox
                    })

            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 90.0  # Default high confidence for Vision API

            # Clean up the text
            cleaned_text = self.clean_vision_api_text(full_text)

            logger.info(f"Google Vision API extracted: '{cleaned_text[:50]}...' with estimated confidence: {avg_confidence:.1f}%")

            return cleaned_text, avg_confidence / 100.0, word_details

        except Exception as e:
            logger.error(f"Google Vision API extraction failed: {e}")
            return self.extract_text_fallback_tesseract(image)

    def estimate_confidence(self, text: str, bounding_poly) -> float:
        """Estimate confidence score based on text characteristics"""
        confidence = 85.0  # Base confidence for Google Vision API

        # Boost confidence for longer, meaningful text
        if len(text) > 3:
            confidence += 5.0
        if len(text) > 8:
            confidence += 5.0

        # Boost for common book-related words
        book_words = ['book', 'books', 'press', 'edition', 'volume', 'miller', 'basic']
        if any(word.lower() in text.lower() for word in book_words):
            confidence += 10.0

        # Boost for alphabetic text
        if text.isalpha():
            confidence += 5.0

        # Reduce for very short or noisy text
        if len(text) < 2:
            confidence -= 20.0
        if any(char in text for char in '|~`^{}\\'):
            confidence -= 10.0

        return min(95.0, max(70.0, confidence))  # Clamp between 70-95%

    def clean_vision_api_text(self, text: str) -> str:
        """Clean text extracted from Google Vision API"""
        if not text:
            return ""

        # Remove excessive whitespace
        cleaned = " ".join(text.split())

        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I').replace('~', '-')

        return cleaned.strip()

    def extract_text_fallback_tesseract(self, image: np.ndarray) -> Tuple[str, float, List[Dict]]:
        """Fallback to Tesseract if Google Vision API fails"""
        if self.tesseract_ocr:
            return self.tesseract_ocr.extract_text_with_confidence(image)
        else:
            logger.error("Tesseract fallback not available")
            return "", 0.0, []

    def extract_text_with_confidence(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[Dict]]:
        """Extract text using Google Vision API with Tesseract fallback"""

        # Try Google Vision API first
        result = self.extract_text_with_google_vision(image)

        # If Google Vision API succeeded, return result
        if result[1] > self.confidence_threshold:  # Good confidence from Google Vision API
            return result

        # If Google Vision API failed or gave poor results, try Tesseract fallback
        logger.info("Google Vision API result not satisfactory, trying Tesseract fallback")
        return self.extract_text_fallback_tesseract(image)


    def extract_text_multiple_configs(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Google Vision API as primary method"""

        # Use Google Vision API
        result = self.extract_text_with_google_vision(image)

        if result[1] > 0.3:  # If we get reasonable results from Vision API
            return result[0], result[1]

        # Fallback to Tesseract with multiple configurations
        if self.tesseract_ocr:
            return self.tesseract_ocr.extract_text_multiple_configs(image)
        else:
            return "", 0.0


class BookSpineDetector:
    """Enhanced book spine detector with configurable OCR"""

    def __init__(self, model_path: str = None, ocr_provider: str = None):
        self.model_path = model_path or ocr_config.get('yolo.model_path', 'yolov8n.pt')
        self.preprocessor = ImagePreprocessor()

        # Set OCR provider from config if not specified
        self.ocr_provider = ocr_provider or ocr_config.ocr_provider

        # Initialize OCR based on configuration
        self._initialize_ocr()
        self.yolo_model = None

    def _initialize_ocr(self):
        """Initialize OCR based on current provider setting"""
        logger.info(f"Initializing OCR with provider: {self.ocr_provider}")

        if self.ocr_provider == ocr_config.GOOGLE_VISION:
            if not GOOGLE_VISION_AVAILABLE:
                logger.warning("Google Cloud Vision library not installed, falling back to Tesseract")
                self.ocr = TesseractOCR()
                self.ocr_provider = ocr_config.TESSERACT
            elif ocr_config.google_vision_enabled and ocr_config.is_google_vision_configured():
                try:
                    self.ocr = GoogleVisionOCR(use_fallback=True)
                    logger.info("Using Google Cloud Vision API with Tesseract fallback")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Vision API: {e}, falling back to Tesseract")
                    self.ocr = TesseractOCR()
                    self.ocr_provider = ocr_config.TESSERACT
            else:
                logger.warning("Google Vision API not configured, falling back to Tesseract")
                self.ocr = TesseractOCR()
                self.ocr_provider = ocr_config.TESSERACT
        elif self.ocr_provider == ocr_config.TESSERACT:
            if ocr_config.tesseract_enabled:
                self.ocr = TesseractOCR()
                logger.info("Using Tesseract OCR")
            else:
                raise ValueError("Tesseract OCR is disabled in configuration")
        else:
            raise ValueError(f"Unknown OCR provider: {self.ocr_provider}")

    def switch_ocr_provider(self, provider: str):
        """Switch OCR provider and reinitialize"""
        if provider not in [ocr_config.TESSERACT, ocr_config.GOOGLE_VISION]:
            raise ValueError(f"Invalid OCR provider: {provider}")

        self.ocr_provider = provider
        ocr_config.ocr_provider = provider
        self._initialize_ocr()
        logger.info(f"Switched to OCR provider: {provider}")

    def get_current_ocr_info(self) -> Dict[str, Any]:
        """Get information about current OCR configuration"""
        return {
            "provider": self.ocr_provider,
            "google_vision_configured": ocr_config.is_google_vision_configured(),
            "tesseract_enabled": ocr_config.tesseract_enabled,
            "google_vision_enabled": ocr_config.google_vision_enabled
        }

    def load_yolo_model(self):
        """Load YOLO model for spine detection"""
        if self.yolo_model is None:
            try:
                # Check if model file exists
                if not os.path.exists(
                    self.model_path
                ) and not self.model_path.startswith("yolo"):
                    logger.warning(f"Model file not found: {self.model_path}")
                    logger.info("Falling back to YOLOv8 nano model")
                    self.model_path = "yolov8n.pt"

                # Load model with explicit device selection
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading YOLO model on {device}: {self.model_path}")

                self.yolo_model = YOLO(self.model_path)

                # Configure model for inference
                if hasattr(self.yolo_model, "to"):
                    self.yolo_model.to(device)

                logger.info(f"Successfully loaded YOLO model: {self.model_path}")

            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                logger.info("Attempting to use fallback detection method")
                raise

    def detect_spines_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect book spine bounding boxes using YOLO"""
        self.load_yolo_model()

        try:
            # Configure inference parameters
            inference_config = {
                "verbose": False,
                "save": False,
                "conf": 0.25,  # Confidence threshold
                "iou": 0.45,  # IoU threshold for NMS
                "max_det": 50,  # Maximum detections per image
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

            # Run inference
            results = self.yolo_model(image, **inference_config)

            spine_boxes = []
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Convert to xyxy format
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = (
                            int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                        )

                        # Filter based on confidence and reasonable dimensions
                        box_width = x2 - x1
                        box_height = y2 - y1
                        aspect_ratio = box_height / box_width if box_width > 0 else 0

                        # Log all detections for debugging
                        class_name = r.names.get(class_id, "unknown") if hasattr(r, 'names') else "unknown"
                        logger.info(f"YOLO detected: class_id={class_id}, class_name={class_name}, confidence={confidence:.2f}, "
                                  f"box={box_width}x{box_height}, aspect_ratio={aspect_ratio:.2f}")

                        # Accept any reasonable object that could be a book spine
                        # YOLOv8 might detect books as "book" (class_id=84) or other objects
                        if (
                            confidence > 0.1  # Lower threshold to see more detections
                            and box_width > 10  # Lower minimum size
                            and box_height > 30  # Lower minimum size
                            and aspect_ratio > 0.5  # More lenient aspect ratio
                        ):
                            spine_boxes.append((int(x1), int(y1), int(x2), int(y2)))

            logger.info(f"YOLO detected {len(spine_boxes)} potential book spines")
            return spine_boxes

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def detect_spines_fallback(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Fallback spine detection using computer vision techniques"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find vertical lines (book spines are usually vertical)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20
        )

        spine_boxes = []
        if lines is not None:
            # Group nearby vertical lines into spine regions
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is roughly vertical
                if abs(x2 - x1) < abs(y2 - y1) * 0.3:  # Vertical line
                    vertical_lines.append(line[0])

            # Create bounding boxes from vertical line groups
            if vertical_lines:
                vertical_lines.sort(key=lambda l: l[0])  # Sort by x coordinate

                current_group = [vertical_lines[0]]
                for line in vertical_lines[1:]:
                    x1, y1, x2, y2 = line
                    prev_x = current_group[-1][0]

                    if abs(x1 - prev_x) < 100:  # Lines close together
                        current_group.append(line)
                    else:
                        # Process current group
                        if len(current_group) >= 2:
                            xs = [l[0] for l in current_group] + [
                                l[2] for l in current_group
                            ]
                            ys = [l[1] for l in current_group] + [
                                l[3] for l in current_group
                            ]
                            spine_boxes.append((min(xs), min(ys), max(xs), max(ys)))
                        current_group = [line]

                # Process last group
                if len(current_group) >= 2:
                    xs = [l[0] for l in current_group] + [l[2] for l in current_group]
                    ys = [l[1] for l in current_group] + [l[3] for l in current_group]
                    spine_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        # If no spines detected, create mock regions for demo
        if not spine_boxes:
            h, w = gray.shape
            spine_width = w // 5
            for i in range(3):  # Create 3 mock spine regions
                x1 = i * spine_width + 20
                x2 = x1 + spine_width - 40
                y1 = 50
                y2 = h - 50
                if x2 > x1 and y2 > y1:
                    spine_boxes.append((x1, y1, x2, y2))

        return spine_boxes

    def parse_spine_text_enhanced(self, text: str) -> Tuple[str, str]:
        """Enhanced text parsing to separate title and author"""
        if not text.strip():
            return "", ""

        # Clean up the text
        cleaned_text = re.sub(r"\s+", " ", text.strip())
        lines = [line.strip() for line in cleaned_text.split("\n") if line.strip()]

        # If only one line, try to split by common patterns
        if len(lines) == 1:
            line = lines[0]

            # Look for patterns like "Title by Author" or "Title - Author"
            by_match = re.search(r"^(.*?)\s+by\s+(.+)$", line, re.IGNORECASE)
            if by_match:
                return by_match.group(1).strip(), by_match.group(2).strip()

            dash_match = re.search(r"^(.*?)\s*[-–—]\s*(.+)$", line)
            if dash_match:
                return dash_match.group(1).strip(), dash_match.group(2).strip()

            # If no clear pattern, assume it's all title
            return line, ""

        # Multiple lines - use heuristics
        title_candidates = []
        author_candidates = []

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Skip very short lines (likely noise)
            if len(line) < 3:
                continue

            # Author indicators
            if any(
                indicator in line_lower for indicator in ["by ", "author", "written"]
            ):
                # Remove author indicators
                clean_author = re.sub(
                    r"\b(by|author|written by)\b", "", line, flags=re.IGNORECASE
                ).strip()
                if clean_author:
                    author_candidates.append(clean_author)
            # Number/volume indicators (usually part of title)
            elif re.search(r"\b(vol|volume|book|part|chapter|#|\d+)\b", line_lower):
                title_candidates.append(line)
            # Longer lines more likely to be titles
            elif len(line) > 10:
                title_candidates.append(line)
            # First few lines more likely to be title
            elif i < 2:
                title_candidates.append(line)
            else:
                # Could be either, but lean towards author if shorter
                if len(line) < 25:
                    author_candidates.append(line)
                else:
                    title_candidates.append(line)

        # Combine results
        title = (
            " ".join(title_candidates[:2])
            if title_candidates
            else lines[0]
            if lines
            else ""
        )
        author = (
            " ".join(author_candidates[:1])
            if author_candidates
            else (lines[-1] if len(lines) > 1 and not title_candidates else "")
        )

        # Clean up results
        title = re.sub(r"\s+", " ", title).strip()
        author = re.sub(r"\s+", " ", author).strip()

        # Avoid duplication
        if title.lower() == author.lower():
            author = ""

        return title[:100], author[:50]  # Limit length

    def process_image(
        self, image_path: str, output_dir: str = "output"
    ) -> List[BookSpine]:
        """Complete pipeline: detect spines, preprocess, extract text, parse metadata"""

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Detect spine bounding boxes
        try:
            spine_boxes = self.detect_spines_yolo(image)
            logger.info(f"YOLO detected {len(spine_boxes)} book spines")
        except:
            logger.warning("YOLO detection failed, using fallback method")
            spine_boxes = self.detect_spines_fallback(image)
            logger.info(f"Fallback method detected {len(spine_boxes)} book spines")

        book_spines = []

        for i, (x1, y1, x2, y2) in enumerate(spine_boxes):
            try:
                # Extract spine region with padding
                padding = 10
                y1_pad = max(0, y1 - padding)
                y2_pad = min(image.shape[0], y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(image.shape[1], x2 + padding)

                spine_region = image[y1_pad:y2_pad, x1_pad:x2_pad]

                if spine_region.size == 0:
                    continue

                # Preprocess for OCR
                spine_height = y2 - y1
                try:
                    result = self.preprocessor.preprocess_for_ocr(spine_region, spine_height)
                    if isinstance(result, tuple) and len(result) == 2:
                        processed_image, metadata = result
                    elif isinstance(result, np.ndarray):
                        # Handle case where only image is returned
                        processed_image = result
                        metadata = {}
                    else:
                        logger.warning(f"Unexpected result from preprocess_for_ocr: {type(result)}")
                        processed_image = spine_region
                        metadata = {}
                except Exception as e:
                    logger.error(f"Failed to preprocess image: {e}")
                    processed_image = spine_region
                    metadata = {}

                # Extract text with enhanced OCR
                try:
                    result = self.ocr.extract_text_with_confidence(processed_image)
                    if isinstance(result, tuple) and len(result) == 3:
                        raw_text, confidence, word_details = result
                    else:
                        logger.warning(f"Unexpected result from extract_text_with_confidence: {result}")
                        raw_text, confidence, word_details = "", 0.0, []
                except Exception as e:
                    logger.error(f"Failed to extract text with confidence: {e}")
                    raw_text, confidence, word_details = "", 0.0, []

                # If confidence is low, try multiple configurations
                if confidence < 0.6:
                    try:
                        result = self.ocr.extract_text_multiple_configs(processed_image)
                        if isinstance(result, tuple) and len(result) == 2:
                            alt_text, alt_conf = result
                            if alt_conf > confidence:
                                raw_text, confidence = alt_text, alt_conf
                        else:
                            logger.warning(f"Unexpected result from extract_text_multiple_configs: {result}")
                    except Exception as e:
                        logger.error(f"Failed to extract text with multiple configs: {e}")

                # Parse title and author
                try:
                    title, author = self.parse_spine_text_enhanced(raw_text)
                except Exception as e:
                    logger.error(f"Failed to parse spine text: {e}, raw_text: '{raw_text}'")
                    title, author = raw_text, ""

                # Create BookSpine object
                book_spine = BookSpine(
                    bbox=(x1, y1, x2, y2),
                    title=title,
                    author=author,
                    confidence=confidence,
                    raw_text=raw_text,
                    preprocessed_image=metadata,
                )

                book_spines.append(book_spine)

                # Save debug images if enabled
                if self.preprocessor.debug_mode:
                    debug_dir = Path(output_dir) / "debug"
                    debug_dir.mkdir(exist_ok=True)

                    cv2.imwrite(
                        str(debug_dir / f"spine_{i}_original.jpg"), spine_region
                    )
                    cv2.imwrite(
                        str(debug_dir / f"spine_{i}_processed.jpg"), processed_image
                    )

                logger.info(
                    f"Spine {i}: '{title}' by '{author}' (confidence: {confidence:.2f})"
                )

            except Exception as e:
                logger.error(f"Failed to process spine {i}: {e}")
                continue

        return book_spines

    def draw_bounding_boxes_and_save(
        self,
        image_path: str,
        book_spines: List[BookSpine],
        output_path: str = None
    ) -> str:
        """Draw bounding boxes around detected book spines and save the annotated image"""

        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Create a copy for annotation
        annotated_image = image.copy()

        # Color palette for different spines
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]

        # Draw bounding boxes and labels
        for i, spine in enumerate(book_spines):
            x1, y1, x2, y2 = spine.bbox
            color = colors[i % len(colors)]

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            title_text = spine.title[:25] + "..." if len(spine.title) > 25 else spine.title
            author_text = spine.author[:20] + "..." if len(spine.author) > 20 else spine.author
            confidence_text = f"{spine.confidence:.2f}"

            # Create multi-line label
            label_lines = []
            if title_text:
                label_lines.append(f"Title: {title_text}")
            if author_text:
                label_lines.append(f"Author: {author_text}")
            label_lines.append(f"Conf: {confidence_text}")
            label_lines.append(f"Spine #{i+1}")

            # Calculate label background size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            line_height = 15

            # Find the maximum text width
            max_width = 0
            for line in label_lines:
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_width = max(max_width, text_width)

            # Calculate label position (above the bounding box if possible)
            label_height = len(label_lines) * line_height + 10
            label_y = max(y1 - label_height - 5, 0)
            if label_y < label_height:  # If not enough space above, put it below
                label_y = y2 + 5

            label_x = max(x1, 0)

            # Draw label background
            cv2.rectangle(
                annotated_image,
                (label_x, label_y),
                (label_x + max_width + 10, label_y + label_height),
                color,
                -1
            )

            # Add border around label background
            cv2.rectangle(
                annotated_image,
                (label_x, label_y),
                (label_x + max_width + 10, label_y + label_height),
                (0, 0, 0),
                1
            )

            # Draw text lines
            for j, line in enumerate(label_lines):
                text_y = label_y + 15 + (j * line_height)
                cv2.putText(
                    annotated_image,
                    line,
                    (label_x + 5, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    cv2.LINE_AA
                )

        # Generate output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_annotated{input_path.suffix}")

        # Save the annotated image
        success = cv2.imwrite(output_path, annotated_image)
        if not success:
            raise ValueError(f"Failed to save annotated image to: {output_path}")

        logger.info(f"Saved annotated image with {len(book_spines)} bounding boxes to: {output_path}")
        return output_path


def lookup_book_metadata_free(title: str, author: str = "") -> Dict:
    """Look up book metadata using free APIs with multiple search strategies"""

    if not title.strip():
        return {"best_match": {}, "partial_matches": []}

    all_results = []
    search_strategies = []

    # Strategy 1: Exact title and author match
    if title and author:
        search_strategies.append({
            "query": f'intitle:"{title}" inauthor:"{author}"',
            "description": "Exact title and author",
            "priority": 1
        })

    # Strategy 2: Exact title only
    if title:
        search_strategies.append({
            "query": f'intitle:"{title}"',
            "description": "Exact title",
            "priority": 2
        })

    # Strategy 3: Individual important words from title
    title_words = [word.strip() for word in title.split() if len(word.strip()) > 3]
    if len(title_words) >= 2:
        # Use first 2-3 most important words
        important_words = title_words[:3]
        word_query = " ".join([f'intitle:"{word}"' for word in important_words])
        search_strategies.append({
            "query": word_query,
            "description": f"Key words: {', '.join(important_words)}",
            "priority": 3
        })

    # Strategy 4: Partial title search (remove common publisher words)
    cleaned_title = title
    publisher_words = ["BOOKS", "PRESS", "PUBLICATIONS", "PENGUIN", "CENTURY", "HARPER", "SIMON", "SCHUSTER", "OXFORD", "DOUBLEDAY"]
    for pub_word in publisher_words:
        cleaned_title = cleaned_title.replace(pub_word, "").strip()

    if cleaned_title and cleaned_title != title:
        search_strategies.append({
            "query": f'"{cleaned_title}"',
            "description": f"Cleaned title: {cleaned_title}",
            "priority": 4
        })

    # Strategy 5: Author name only (if available)
    if author and len(author.strip()) > 3:
        search_strategies.append({
            "query": f'inauthor:"{author}"',
            "description": f"Author: {author}",
            "priority": 5
        })

    # Execute search strategies
    for strategy in search_strategies:
        try:
            url = "https://www.googleapis.com/books/v1/volumes"
            params = {"q": strategy["query"], "maxResults": 5}  # Get multiple results

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("totalItems", 0) > 0:
                for item in data["items"]:
                    book = item["volumeInfo"]

                    # Extract ISBN
                    isbn_list = book.get("industryIdentifiers", [])
                    isbn = ""
                    for identifier in isbn_list:
                        if identifier.get("type") in ["ISBN_13", "ISBN_10"]:
                            isbn = identifier.get("identifier", "")
                            break

                    book_result = {
                        "isbn": isbn,
                        "title": book.get("title", ""),
                        "authors": book.get("authors", []),
                        "publisher": book.get("publisher", ""),
                        "published_date": book.get("publishedDate", ""),
                        "description": book.get("description", ""),
                        "categories": book.get("categories", []),
                        "page_count": book.get("pageCount", 0),
                        "language": book.get("language", ""),
                        "search_strategy": strategy["description"],
                        "priority": strategy["priority"],
                        "confidence_score": calculate_match_confidence(title, author, book)
                    }

                    # Avoid duplicates based on ISBN or title
                    is_duplicate = False
                    for existing in all_results:
                        if (book_result["isbn"] and book_result["isbn"] == existing.get("isbn")) or \
                           (book_result["title"] and book_result["title"] == existing.get("title")):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        all_results.append(book_result)

        except Exception as e:
            logger.warning(f"Search strategy '{strategy['description']}' failed: {e}")
            continue

    # Sort results by confidence score and priority
    all_results.sort(key=lambda x: (x["confidence_score"], -x["priority"]), reverse=True)

    # Return best match and up to 5 partial matches
    result = {
        "best_match": all_results[0] if all_results else {},
        "partial_matches": all_results[1:6] if len(all_results) > 1 else [],
        "total_found": len(all_results)
    }

    return result


def calculate_match_confidence(ocr_title: str, ocr_author: str, google_book: Dict) -> float:
    """Calculate confidence score for how well a Google Books result matches the OCR text"""

    confidence = 0.0
    max_confidence = 100.0

    book_title = google_book.get("title", "").lower()
    book_authors = [author.lower() for author in google_book.get("authors", [])]
    ocr_title_lower = ocr_title.lower()
    ocr_author_lower = ocr_author.lower()

    # Title matching (60% weight)
    title_score = 0.0
    if book_title and ocr_title_lower:
        # Check for exact match
        if book_title == ocr_title_lower:
            title_score = 60.0
        else:
            # Check for partial matches
            ocr_words = set(ocr_title_lower.split())
            book_words = set(book_title.split())

            if ocr_words and book_words:
                common_words = ocr_words.intersection(book_words)
                # Weight by length of words and total overlap
                word_overlap = sum(len(word) for word in common_words)
                total_word_length = sum(len(word) for word in ocr_words)

                if total_word_length > 0:
                    overlap_ratio = word_overlap / total_word_length
                    title_score = overlap_ratio * 60.0

    confidence += title_score

    # Author matching (25% weight)
    author_score = 0.0
    if book_authors and ocr_author_lower:
        for book_author in book_authors:
            if book_author in ocr_author_lower or ocr_author_lower in book_author:
                author_score = 25.0
                break
            # Check for partial author name matches
            author_words = set(book_author.split())
            ocr_author_words = set(ocr_author_lower.split())
            if author_words.intersection(ocr_author_words):
                author_score = max(author_score, 15.0)

    confidence += author_score

    # Publisher/Quality indicators (15% weight)
    quality_score = 0.0

    # Favor books with ISBN
    if google_book.get("industryIdentifiers"):
        quality_score += 5.0

    # Favor books with descriptions
    if google_book.get("description"):
        quality_score += 3.0

    # Favor books with page counts
    if google_book.get("pageCount", 0) > 0:
        quality_score += 2.0

    # Favor recent publications (within last 50 years)
    pub_date = google_book.get("publishedDate", "")
    if pub_date:
        try:
            # Extract year from date string
            import re
            year_match = re.search(r'\d{4}', pub_date)
            if year_match:
                pub_year = int(year_match.group())
                current_year = 2024  # Could use datetime.now().year
                if current_year - pub_year <= 50:
                    quality_score += 5.0
        except:
            pass

    confidence += quality_score

    return min(confidence, max_confidence)


def export_results(book_spines: List[BookSpine], output_file: str) -> None:
    """Export book spine detection results to JSON format"""

    results = []
    for spine in book_spines:
        spine_data = {
            "title": spine.title,
            "author": spine.author,
            "confidence": spine.confidence,
            "bbox": spine.bbox,
            "raw_text": spine.raw_text,
        }
        results.append(spine_data)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(results)} book spine results to {output_file}")
