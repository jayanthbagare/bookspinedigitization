#!/usr/bin/env python3
"""
Configuration settings for Book Spine Digitization
"""

import os
from typing import Dict, Any
import json
from pathlib import Path

class OCRConfig:
    """OCR Configuration settings"""

    # OCR Provider options
    TESSERACT = "tesseract"
    GOOGLE_VISION = "google_vision"

    # Default settings
    DEFAULT_PROVIDER = GOOGLE_VISION

    def __init__(self, config_file: str = "ocr_config.json"):
        self.config_file = config_file
        self.settings = self._load_settings()

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from config file or create defaults"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                return self._get_default_settings()
        else:
            settings = self._get_default_settings()
            self._save_settings(settings)
            return settings

    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default configuration settings"""
        return {
            "ocr_provider": self.DEFAULT_PROVIDER,
            "google_vision": {
                "enabled": True,
                "use_fallback": True,
                "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
                "project_id": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                "confidence_threshold": 0.5
            },
            "tesseract": {
                "enabled": True,
                "confidence_threshold": 60,
                "psm": 6,
                "oem": 1,
                "char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-()[]&* "
            },
            "preprocessing": {
                "debug_mode": False,
                "min_spine_height": 50,
                "min_spine_width": 20,
                "min_aspect_ratio": 1.5,
                "upscale_factor": 2,
                "target_height": 600
            },
            "yolo": {
                "model_path": "yolov8n.pt",
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "max_detections": 50
            }
        }

    def _save_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config file {self.config_file}: {e}")

    def get(self, key: str, default=None):
        """Get a setting value"""
        keys = key.split('.')
        value = self.settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value) -> None:
        """Set a setting value and save"""
        keys = key.split('.')
        settings = self.settings

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in settings:
                settings[k] = {}
            settings = settings[k]

        # Set the value
        settings[keys[-1]] = value
        self._save_settings(self.settings)

    @property
    def ocr_provider(self) -> str:
        """Get current OCR provider"""
        return self.get('ocr_provider', self.DEFAULT_PROVIDER)

    @ocr_provider.setter
    def ocr_provider(self, provider: str) -> None:
        """Set OCR provider"""
        if provider not in [self.TESSERACT, self.GOOGLE_VISION]:
            raise ValueError(f"Invalid OCR provider: {provider}")
        self.set('ocr_provider', provider)

    @property
    def google_vision_enabled(self) -> bool:
        """Check if Google Vision API is enabled"""
        return self.get('google_vision.enabled', True)

    @property
    def tesseract_enabled(self) -> bool:
        """Check if Tesseract is enabled"""
        return self.get('tesseract.enabled', True)

    @property
    def google_credentials_path(self) -> str:
        """Get Google Cloud credentials path"""
        return self.get('google_vision.credentials_path', '')

    def is_google_vision_configured(self) -> bool:
        """Check if Google Vision API is properly configured"""
        credentials = self.google_credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        return bool(credentials and Path(credentials).exists())

    def print_current_settings(self) -> None:
        """Print current OCR settings"""
        print("=== OCR Configuration ===")
        print(f"OCR Provider: {self.ocr_provider}")
        print(f"Google Vision Enabled: {self.google_vision_enabled}")
        print(f"Google Vision Configured: {self.is_google_vision_configured()}")
        print(f"Tesseract Enabled: {self.tesseract_enabled}")
        print(f"Google Credentials: {self.google_credentials_path or 'From environment'}")
        print("========================")

    def switch_to_tesseract(self) -> None:
        """Switch to Tesseract OCR"""
        self.ocr_provider = self.TESSERACT
        print("Switched to Tesseract OCR")

    def switch_to_google_vision(self) -> None:
        """Switch to Google Vision API"""
        if not self.is_google_vision_configured():
            print("Warning: Google Vision API not properly configured!")
            print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("or update the credentials_path in the config file")
        self.ocr_provider = self.GOOGLE_VISION
        print("Switched to Google Vision API")

# Global config instance
ocr_config = OCRConfig()