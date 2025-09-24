# OCR Configuration Guide

This project now supports two OCR providers with easy switching between them:

## OCR Providers

### 1. Google Cloud Vision API (Recommended for accuracy)
- Higher accuracy for text recognition
- Better handling of skewed or distorted text
- Requires Google Cloud credentials
- Uses Tesseract as fallback

### 2. Tesseract OCR (Local processing)
- No external dependencies or API calls
- Fast local processing
- Good for simple, clear text
- Already included in the project

## Quick Setup

### Option 1: Use the Interactive Settings Tool
```bash
python ocr_settings.py
```

### Option 2: Command Line Options
```bash
# Switch to Google Vision API
python main.py --ocr google_vision

# Switch to Tesseract
python main.py --ocr tesseract

# Show current configuration
python main.py --config
```

## Google Cloud Vision API Setup

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Vision API

### Step 2: Create Service Account
1. Go to IAM & Admin > Service Accounts
2. Create a new service account
3. Download the JSON key file

### Step 3: Set Environment Variable
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### Step 4: Test Configuration
```bash
python ocr_settings.py --status
```

## Configuration Files

The system creates an `ocr_config.json` file to store your settings:

```json
{
  "ocr_provider": "google_vision",
  "google_vision": {
    "enabled": true,
    "use_fallback": true,
    "credentials_path": "",
    "confidence_threshold": 0.5
  },
  "tesseract": {
    "enabled": true,
    "confidence_threshold": 60,
    "psm": 6,
    "oem": 1
  }
}
```

## Usage Examples

### Basic Usage (Uses configured provider)
```bash
python main.py
```

### Force specific provider
```bash
# Use Google Vision API
python main.py --ocr google_vision

# Use Tesseract only
python main.py --ocr tesseract
```

### Check current settings
```bash
python main.py --config
```

### Interactive configuration
```bash
python ocr_settings.py
```

## Troubleshooting

### Google Vision API Issues
- **"Credentials not found"**: Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- **"Permission denied"**: Ensure Vision API is enabled in Google Cloud Console
- **"Quota exceeded"**: Check your Google Cloud billing and quotas

### Tesseract Issues
- **"Tesseract not found"**: Install Tesseract OCR (`apt-get install tesseract-ocr`)
- **Poor accuracy**: Adjust confidence threshold or PSM mode in settings

### General Issues
- Check `ocr_config.json` is readable and valid JSON
- Ensure image files exist and are readable
- Check console output for detailed error messages

## Performance Comparison

| Feature | Google Vision API | Tesseract |
|---------|------------------|-----------|
| Accuracy | Higher | Good |
| Speed | Network dependent | Fast |
| Cost | Pay per use | Free |
| Setup | Requires credentials | Simple |
| Offline | No | Yes |

## Default Behavior

- **Primary**: Google Vision API (if configured)
- **Fallback**: Tesseract OCR
- **Auto-switching**: Falls back to Tesseract if Google Vision fails
- **Confidence thresholds**: Configurable per provider