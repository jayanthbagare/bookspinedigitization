# Book Spine Digitization

A Python tool for detecting and reading book spines from images using computer vision and OCR.

## Features

- **Dual OCR Support**: Google Cloud Vision API and Tesseract OCR with easy switching
- **YOLO Object Detection**: Automatic book spine detection in images
- **Advanced Preprocessing**: Image enhancement for better OCR accuracy
- **Metadata Enrichment**: Google Books API integration for additional book information
- **Configurable Settings**: Easy configuration management with JSON config files
- **Interactive CLI**: Command-line tools for configuration and processing

## Quick Start

### 1. Install Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng libgl1-mesa-glx libglib2.0-0
```

#### macOS
```bash
brew install tesseract
```

### 2. Install Python Requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
# Process image with default settings
python main.py

# Show OCR configuration options
python main.py --help
```

## OCR Configuration

### Quick OCR Setup
```bash
# Interactive configuration tool
python ocr_settings.py

# Command line options
python main.py --ocr tesseract        # Use Tesseract OCR
python main.py --ocr google_vision    # Use Google Vision API
python main.py --config               # Show current settings
```

### Google Cloud Vision API Setup (Optional)
1. Create a Google Cloud project and enable Vision API
2. Create a service account and download the JSON key
3. Set environment variable:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

See [OCR_SETUP.md](OCR_SETUP.md) for detailed configuration instructions.

## Usage Examples

### Basic Processing
```python
from lib import BookSpineDetector

# Initialize detector (uses configured OCR provider)
detector = BookSpineDetector()

# Process bookshelf image
book_spines = detector.process_image("your_bookshelf.jpg")

# Print results
for spine in book_spines:
    print(f"Title: {spine.title}")
    print(f"Author: {spine.author}")
    print(f"Confidence: {spine.confidence:.2f}")
```

### Switch OCR Provider
```python
from lib import BookSpineDetector

# Force specific OCR provider
detector = BookSpineDetector(ocr_provider="tesseract")

# Or switch at runtime
detector.switch_ocr_provider("google_vision")
```

## File Structure

```
bookspinedigitization/
├── main.py              # Main processing script
├── lib.py               # Core detection and OCR classes
├── config.py            # Configuration management
├── ocr_settings.py      # Interactive settings tool
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── OCR_SETUP.md        # Detailed OCR configuration guide
└── models/             # YOLO model weights (auto-downloaded)
```

## Command Line Interface

```bash
# Basic usage
python main.py                        # Process with default settings

# OCR provider selection
python main.py --ocr tesseract        # Force Tesseract
python main.py --ocr google_vision    # Force Google Vision API

# Configuration
python main.py --config               # Show current config
python main.py --help                 # Show help

# Settings management
python ocr_settings.py                # Interactive configuration
python ocr_settings.py --status       # Show settings
python ocr_settings.py --google       # Switch to Google Vision
python ocr_settings.py --tesseract    # Switch to Tesseract
```
