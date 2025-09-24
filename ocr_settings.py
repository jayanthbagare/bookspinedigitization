#!/usr/bin/env python3
"""
OCR Settings Management Tool for Book Spine Digitization
"""

import sys
import os
from pathlib import Path
from config import ocr_config

def print_header():
    """Print application header"""
    print("=" * 50)
    print("Book Spine Digitization - OCR Settings")
    print("=" * 50)

def show_current_settings():
    """Show current OCR settings"""
    print("\n=== Current OCR Settings ===")
    print(f"OCR Provider: {ocr_config.ocr_provider}")
    print(f"Google Vision Enabled: {ocr_config.google_vision_enabled}")
    print(f"Google Vision Configured: {ocr_config.is_google_vision_configured()}")
    print(f"Tesseract Enabled: {ocr_config.tesseract_enabled}")
    print(f"Configuration File: {ocr_config.config_file}")

    # Google Vision API settings
    print("\n--- Google Vision API Settings ---")
    print(f"Credentials Path: {ocr_config.google_credentials_path or 'From environment'}")
    print(f"Environment Variable: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
    print(f"Confidence Threshold: {ocr_config.get('google_vision.confidence_threshold', 0.5)}")

    # Tesseract settings
    print("\n--- Tesseract Settings ---")
    print(f"Confidence Threshold: {ocr_config.get('tesseract.confidence_threshold', 60)}")
    print(f"PSM Mode: {ocr_config.get('tesseract.psm', 6)}")
    print(f"OEM Mode: {ocr_config.get('tesseract.oem', 1)}")

    print("=" * 30)

def switch_provider():
    """Interactive OCR provider switching"""
    print("\n=== Switch OCR Provider ===")
    print("Available providers:")
    print("1. Google Cloud Vision API")
    print("2. Tesseract OCR")
    print("3. Cancel")

    choice = input("\nSelect provider (1-3): ").strip()

    if choice == "1":
        if not ocr_config.is_google_vision_configured():
            print("\nWARNING: Google Vision API is not properly configured!")
            print("You need to set up Google Cloud credentials first.")
            print("Set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key file.")

            confirm = input("Do you want to switch anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return

        ocr_config.switch_to_google_vision()
        print("\n✓ Switched to Google Cloud Vision API")

    elif choice == "2":
        ocr_config.switch_to_tesseract()
        print("\n✓ Switched to Tesseract OCR")

    elif choice == "3":
        print("Cancelled.")
        return

    else:
        print("Invalid choice. Please select 1, 2, or 3.")

def configure_google_vision():
    """Configure Google Vision API settings"""
    print("\n=== Configure Google Vision API ===")

    current_path = ocr_config.google_credentials_path
    env_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')

    print(f"Current credentials path: {current_path or 'Not set'}")
    print(f"Environment variable: {env_path or 'Not set'}")

    print("\nOptions:")
    print("1. Set credentials file path")
    print("2. Use environment variable (recommended)")
    print("3. Test current configuration")
    print("4. Back to main menu")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        path = input("Enter path to service account key file: ").strip()
        if path and Path(path).exists():
            ocr_config.set('google_vision.credentials_path', path)
            print(f"✓ Credentials path set to: {path}")
        else:
            print("Error: File not found or invalid path.")

    elif choice == "2":
        print("\nTo use environment variable:")
        print("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json")
        print("Then restart the application.")

    elif choice == "3":
        if ocr_config.is_google_vision_configured():
            print("✓ Google Vision API appears to be configured correctly.")
            try:
                from google.cloud import vision
                client = vision.ImageAnnotatorClient()
                print("✓ Successfully initialized Vision API client.")
            except Exception as e:
                print(f"✗ Error initializing client: {e}")
        else:
            print("✗ Google Vision API is not configured.")

    elif choice == "4":
        return
    else:
        print("Invalid choice.")

def configure_tesseract():
    """Configure Tesseract settings"""
    print("\n=== Configure Tesseract ===")

    print(f"Current confidence threshold: {ocr_config.get('tesseract.confidence_threshold', 60)}")
    print(f"Current PSM mode: {ocr_config.get('tesseract.psm', 6)}")
    print(f"Current OEM mode: {ocr_config.get('tesseract.oem', 1)}")

    print("\nOptions:")
    print("1. Set confidence threshold (0-100)")
    print("2. Set PSM mode (Page Segmentation Mode)")
    print("3. Test Tesseract installation")
    print("4. Back to main menu")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        try:
            threshold = int(input("Enter confidence threshold (0-100): "))
            if 0 <= threshold <= 100:
                ocr_config.set('tesseract.confidence_threshold', threshold)
                print(f"✓ Confidence threshold set to: {threshold}")
            else:
                print("Error: Threshold must be between 0 and 100.")
        except ValueError:
            print("Error: Please enter a valid number.")

    elif choice == "2":
        print("\nCommon PSM modes:")
        print("  6 - Uniform block of text (default)")
        print("  7 - Single text line")
        print("  8 - Single word")
        print("  13 - Raw line")

        try:
            psm = int(input("Enter PSM mode: "))
            if 0 <= psm <= 13:
                ocr_config.set('tesseract.psm', psm)
                print(f"✓ PSM mode set to: {psm}")
            else:
                print("Error: PSM mode must be between 0 and 13.")
        except ValueError:
            print("Error: Please enter a valid number.")

    elif choice == "3":
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"✓ Tesseract is installed: {version}")
        except Exception as e:
            print(f"✗ Tesseract test failed: {e}")

    elif choice == "4":
        return
    else:
        print("Invalid choice.")

def main_menu():
    """Main menu loop"""
    while True:
        print_header()
        show_current_settings()

        print("\n=== Main Menu ===")
        print("1. Switch OCR Provider")
        print("2. Configure Google Vision API")
        print("3. Configure Tesseract")
        print("4. Reset to Defaults")
        print("5. Exit")

        choice = input("\nSelect option (1-5): ").strip()

        if choice == "1":
            switch_provider()
        elif choice == "2":
            configure_google_vision()
        elif choice == "3":
            configure_tesseract()
        elif choice == "4":
            confirm = input("Reset all settings to defaults? (y/N): ").strip().lower()
            if confirm == 'y':
                # Recreate config with defaults
                try:
                    os.remove(ocr_config.config_file)
                    print("✓ Settings reset to defaults.")
                    # Reload config
                    ocr_config.settings = ocr_config._load_settings()
                except Exception as e:
                    print(f"Error resetting settings: {e}")
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-5.")

        input("\nPress Enter to continue...")

def print_usage():
    """Print command line usage"""
    print("OCR Settings Management Tool")
    print("\nUsage:")
    print("  python ocr_settings.py              # Interactive menu")
    print("  python ocr_settings.py --status     # Show current settings")
    print("  python ocr_settings.py --google     # Switch to Google Vision")
    print("  python ocr_settings.py --tesseract  # Switch to Tesseract")
    print("  python ocr_settings.py --help       # Show this help")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if "--help" in sys.argv:
            print_usage()
        elif "--status" in sys.argv:
            print_header()
            show_current_settings()
        elif "--google" in sys.argv:
            ocr_config.switch_to_google_vision()
            print("Switched to Google Vision API")
        elif "--tesseract" in sys.argv:
            ocr_config.switch_to_tesseract()
            print("Switched to Tesseract OCR")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        main_menu()