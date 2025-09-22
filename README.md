# Bookworm - Book spine Digitization

### Install all the OS level Dependencies

#### For Ubuntu/Debian

```
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
sudo apt install libgl1-mesa-glx libglib2.0-0
```

#### For macOS

```
brew install tesseract
```

### Install the requirements by running

```
pip install - r requirements.txt
```

## Setup Instructions

### Clone and setup project

```
git clone https://github.com/jayanthbagare/bookspinedigitization.git
cd bookspinedigitization
pip install -r requirements.txt
```

### Download YOLO Weights

```
# Create models directory
mkdir -p models/yolo_weights

#### Download the trained weights (replace with actual download link)
#### Place best.pt in models/yolo_weights/
```

### Basic Usage

```
from enhanced_book_spine_detector import EnhancedBookSpineDetector

# Initialize detector
detector = EnhancedBookSpineDetector()

# Process bookshelf image
book_spines = detector.process_image("your_bookshelf.jpg")

# Print results
for spine in book_spines:
    print(f"Title: {spine.title}")
    print(f"Author: {spine.author}")
    print(f"Confidence: {spine.confidence:.2f}")
```
