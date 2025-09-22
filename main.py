#!/usr/bin/env python3
"""
Enhanced Book Spine Detection with Automatic Image Inversion
Modified from MinHanLiWesley/book-spine-recognition to include intelligent preprocessing
"""

import cv2
import numpy as np
import os
import json
import requests
from PIL import Image, ImageEnhance
import pytesseract
from google.cloud import vision
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

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
    preprocessed_image: Optional[np.ndarray] = None


class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR results"""

    def __init__(self):
        self.debug_mode = False

    def detect_text_polarity(self, image: np.ndarray, sample_regions: int = 5) -> bool:
        """
        Detect if text is light-on-dark (needs inversion) or dark-on-light
        Returns True if image needs inversion (light text on dark background)
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        h, w = gray.shape

        # Sample multiple regions to get better assessment
        scores = []

        for i in range(sample_regions):
            # Get random sample region
            y1 = np.random.randint(0, h // 3)
            y2 = np.random.randint(2 * h // 3, h)
            x1 = np.random.randint(0, w // 4)
            x2 = np.random.randint(3 * w // 4, w)

            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                continue

            # Calculate average intensity
            avg_intensity = np.mean(region)

            # Apply basic thresholding
            _, binary = cv2.threshold(
                region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Count white vs black pixels
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)

            # If more white pixels and low average intensity -> likely light text on dark bg
            if white_pixels < black_pixels and avg_intensity < 127:
                scores.append(1)  # Needs inversion
            else:
                scores.append(0)  # Normal polarity

        # Return True if majority of samples suggest inversion is needed
        return sum(scores) > len(scores) / 2

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def preprocess_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Complete preprocessing pipeline for optimal OCR
        Returns processed image and metadata about transformations applied
        """
        metadata = {
            "inverted": False,
            "enhanced_contrast": False,
            "denoised": False,
            "border_added": False,
        }

        # Step 1: Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Step 2: Check if inversion is needed
        needs_inversion = self.detect_text_polarity(image)
        if needs_inversion:
            gray = cv2.bitwise_not(gray)
            metadata["inverted"] = True
            logger.info("Applied image inversion for light-text-on-dark")

        # Step 3: Enhance contrast
        gray = self.enhance_contrast(gray)
        metadata["enhanced_contrast"] = True

        # Step 4: Denoise
        gray = cv2.medianBlur(gray, 3)
        metadata["denoised"] = True

        # Step 5: Add white border (crucial for Tesseract)
        border_size = 10
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

        # Step 6: Final binarization with optimal thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary, metadata


class EnhancedBookSpineDetector:
    """Enhanced book spine detector with automatic preprocessing"""

    def __init__(self, model_path: str = "models/yolo_weights/best.pt"):
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.vision_client = vision.ImageAnnotatorClient()

    def detect_spines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect book spine bounding boxes using YOLO
        This is a placeholder - you'd integrate with the actual YOLO model
        """
        # Placeholder implementation
        # In real implementation, load YOLO model and run detection
        h, w = image.shape[:2]

        # Mock detection results for demonstration
        mock_detections = [
            (50, 100, 150, 500),  # x1, y1, x2, y2
            (160, 100, 260, 500),
            (270, 100, 370, 500),
        ]

        return mock_detections

    def extract_text_ocr(
        self, spine_image: np.ndarray, use_cloud_vision: bool = True
    ) -> Dict:
        """Extract text using OCR with preprocessing"""

        # Preprocess image for optimal OCR
        processed_image, metadata = self.preprocessor.preprocess_for_ocr(spine_image)

        extracted_text = ""
        confidence = 0.0

        if use_cloud_vision:
            try:
                # Use Google Cloud Vision API
                extracted_text, confidence = self._extract_with_cloud_vision(
                    processed_image
                )
            except Exception as e:
                logger.warning(f"Cloud Vision failed: {e}, falling back to Tesseract")
                use_cloud_vision = False

        if not use_cloud_vision:
            # Fallback to Tesseract with optimized settings
            extracted_text, confidence = self._extract_with_tesseract(processed_image)

        return {
            "text": extracted_text,
            "confidence": confidence,
            "preprocessing_metadata": metadata,
        }

    def _extract_with_cloud_vision(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Google Cloud Vision API"""
        # Convert numpy array to bytes
        _, buffer = cv2.imencode(".png", image)
        image_bytes = buffer.tobytes()

        image = vision.Image(content=image_bytes)
        response = self.vision_client.text_detection(image=image)

        if response.text_annotations:
            text = response.text_annotations[0].description
            # Calculate average confidence
            confidence = sum(
                [
                    vertex.confidence
                    for vertex in response.text_annotations
                    if hasattr(vertex, "confidence")
                ]
            ) / len(response.text_annotations)
            return text.strip(), confidence

        return "", 0.0

    def _extract_with_tesseract(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Tesseract with optimized configuration"""

        # Tesseract configuration optimized for book spines
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;"\'-()& '

        try:
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)

            # Get confidence data
            data = pytesseract.image_to_data(
                image, config=custom_config, output_type=pytesseract.Output.DICT
            )
            confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return text.strip(), avg_confidence / 100.0

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return "", 0.0

    def parse_spine_text(self, text: str) -> Tuple[str, str]:
        """
        Parse extracted text to separate title and author
        This is a simplified implementation - could be enhanced with NLP
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return "", ""

        if len(lines) == 1:
            # Single line - assume it's the title
            return lines[0], ""

        # Multiple lines - heuristic approach
        # Typically author is at bottom, title at top
        title_candidates = []
        author_candidates = []

        for line in lines:
            # Common author indicators
            if any(word in line.lower() for word in ["by", "author", "written"]):
                author_candidates.append(line)
            # Title usually longer and contains more content words
            elif len(line) > 10:
                title_candidates.append(line)
            else:
                # Shorter lines could be either
                if len(title_candidates) == 0:
                    title_candidates.append(line)
                else:
                    author_candidates.append(line)

        title = " ".join(title_candidates) if title_candidates else lines[0]
        author = (
            " ".join(author_candidates)
            if author_candidates
            else (lines[-1] if len(lines) > 1 else "")
        )

        return title, author

    def process_image(
        self, image_path: str, output_dir: str = "output"
    ) -> List[BookSpine]:
        """
        Complete pipeline: detect spines, preprocess, extract text, parse metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Detect spine bounding boxes
        spine_boxes = self.detect_spines(image)
        logger.info(f"Detected {len(spine_boxes)} book spines")

        book_spines = []

        for i, (x1, y1, x2, y2) in enumerate(spine_boxes):
            # Extract spine region
            spine_region = image[y1:y2, x1:x2]

            # Extract text with preprocessing
            ocr_result = self.extract_text_ocr(spine_region)

            # Parse title and author
            title, author = self.parse_spine_text(ocr_result["text"])

            # Create BookSpine object
            book_spine = BookSpine(
                bbox=(x1, y1, x2, y2),
                title=title,
                author=author,
                confidence=ocr_result["confidence"],
                preprocessed_image=ocr_result.get("preprocessing_metadata"),
            )

            book_spines.append(book_spine)

            # Save debug images if needed
            if self.preprocessor.debug_mode:
                debug_path = os.path.join(output_dir, f"spine_{i}_debug.png")
                cv2.imwrite(debug_path, spine_region)

            logger.info(
                f"Spine {i}: '{title}' by '{author}' (confidence: {ocr_result['confidence']:.2f})"
            )

        return book_spines


def lookup_book_metadata(title: str, author: str = "") -> Dict:
    """
    Look up book metadata using Google Books API
    """
    # Construct search query
    query_parts = []
    if title:
        query_parts.append(f'intitle:"{title}"')
    if author:
        query_parts.append(f'inauthor:"{author}"')

    query = " ".join(query_parts)

    # Google Books API request
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("totalItems", 0) > 0:
            book = data["items"][0]["volumeInfo"]
            return {
                "title": book.get("title", ""),
                "authors": book.get("authors", []),
                "publisher": book.get("publisher", ""),
                "publishedDate": book.get("publishedDate", ""),
                "isbn": book.get("industryIdentifiers", []),
                "description": book.get("description", ""),
                "pageCount": book.get("pageCount", 0),
                "categories": book.get("categories", []),
                "thumbnail": book.get("imageLinks", {}).get("thumbnail", ""),
            }
    except Exception as e:
        logger.error(f"Failed to lookup book metadata: {e}")

    return {}


def main():
    """Example usage of the enhanced book spine detection system"""

    # Initialize detector
    detector = EnhancedBookSpineDetector()
    detector.preprocessor.debug_mode = True  # Enable debug output

    # Process image
    image_path = "sample_bookshelf.jpg"  # Replace with your image path

    try:
        book_spines = detector.process_image(image_path)

        # Display results and lookup metadata
        print("\n=== DETECTED BOOK SPINES ===")
        for i, spine in enumerate(book_spines):
            print(f"\nBook {i + 1}:")
            print(f"  Title: {spine.title}")
            print(f"  Author: {spine.author}")
            print(f"  Confidence: {spine.confidence:.2f}")
            print(f"  Bounding Box: {spine.bbox}")

            # Look up additional metadata
            if spine.title:
                metadata = lookup_book_metadata(spine.title, spine.author)
                if metadata:
                    print(f"  Publisher: {metadata.get('publisher', 'N/A')}")
                    print(f"  Published: {metadata.get('publishedDate', 'N/A')}")
                    if metadata.get("isbn"):
                        isbn_list = [
                            isbn.get("identifier", "") for isbn in metadata["isbn"]
                        ]
                        print(f"  ISBN: {', '.join(isbn_list)}")

        # Save results to JSON
        results = []
        for spine in book_spines:
            metadata = lookup_book_metadata(spine.title, spine.author)
            results.append(
                {
                    "bbox": spine.bbox,
                    "title": spine.title,
                    "author": spine.author,
                    "confidence": spine.confidence,
                    "metadata": metadata,
                }
            )

        with open("detected_books.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to detected_books.json")
        print(f"Found {len(book_spines)} books total")

    except Exception as e:
        logger.error(f"Processing failed: {e}")


if __name__ == "__main__":
    main()
