from lib import BookSpineDetector, export_results, lookup_book_metadata_free
from config import ocr_config
import re
import sys


def clean_title(title):
    """Clean up OCR errors in title for better API search"""
    if not title:
        return ""

    # Remove common OCR noise
    cleaned = re.sub(r"[^\w\s\-:.,]", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove very short words that are likely OCR errors
    words = [
        word
        for word in cleaned.split()
        if len(word) > 2 or word.upper() in ["EX", "AI", "OF", "OR", "IN", "ON"]
    ]

    return " ".join(words)


def print_usage():
    """Print usage instructions"""
    print("\nUsage: python main.py [options]")
    print("\nOptions:")
    print("  --ocr <provider>    Set OCR provider (google_vision or tesseract)")
    print("  --config           Show current configuration")
    print("  --help             Show this help message")
    print("\nOCR Providers:")
    print("  google_vision      Use Google Cloud Vision API (requires credentials)")
    print("  tesseract          Use Tesseract OCR (local processing)")
    print("\nExamples:")
    print("  python main.py                        # Use configured OCR provider")
    print("  python main.py --ocr tesseract        # Force use Tesseract")
    print("  python main.py --ocr google_vision    # Force use Google Vision API")
    print("  python main.py --config               # Show current settings")


# Parse command line arguments
if len(sys.argv) > 1:
    if "--help" in sys.argv:
        print_usage()
        sys.exit(0)
    elif "--config" in sys.argv:
        ocr_config.print_current_settings()
        sys.exit(0)
    elif "--ocr" in sys.argv:
        try:
            ocr_index = sys.argv.index("--ocr")
            if ocr_index + 1 < len(sys.argv):
                ocr_provider = sys.argv[ocr_index + 1]
                if ocr_provider == "google_vision":
                    ocr_config.switch_to_google_vision()
                elif ocr_provider == "tesseract":
                    ocr_config.switch_to_tesseract()
                else:
                    print(f"Error: Invalid OCR provider '{ocr_provider}'")
                    print("Valid options are: google_vision, tesseract")
                    sys.exit(1)
            else:
                print("Error: --ocr requires a provider argument")
                print_usage()
                sys.exit(1)
        except ValueError:
            print("Error: Invalid --ocr usage")
            print_usage()
            sys.exit(1)

# Print current OCR configuration
print("=== OCR Configuration ===")
print(f"OCR Provider: {ocr_config.ocr_provider}")
print(f"Google Vision Configured: {ocr_config.is_google_vision_configured()}")
if (
    ocr_config.ocr_provider == ocr_config.GOOGLE_VISION
    and not ocr_config.is_google_vision_configured()
):
    print("WARNING: Google Vision API selected but not properly configured!")
    print("Please set GOOGLE_APPLICATION_CREDENTIALS environment variable")
print("========================\n")

# Initialize detector
detector = BookSpineDetector()

# Process image
image_file = "b3.jpg"
book_spines = detector.process_image(image_file, "results/")

# Draw bounding boxes and save annotated image
if book_spines:
    annotated_image_path = detector.draw_bounding_boxes_and_save(
        image_file,
        book_spines,
        f"results/{image_file.split('.')[0]}_with_bounding_boxes.jpg",
    )
    print(f"\nAnnotated image saved: {annotated_image_path}")

# Enhanced results with Google Books API
enhanced_results = []

print("=== BOOK SPINE DETECTION RESULTS ===\n")

for i, spine in enumerate(book_spines, 1):
    print(f"Book {i}:")
    print(f"  Title: {spine.title}")
    print(f"  Author: {spine.author}")
    print(f"  Confidence: {spine.confidence:.2f}")
    print(f"  Bounding box: {spine.bbox}")

    # Clean title for better API search
    clean_title_text = clean_title(spine.title)
    clean_author_text = clean_title(spine.author)

    # Get additional metadata from Google Books API
    metadata_result = {}
    if clean_title_text and len(clean_title_text) > 3:
        print(f"  Searching Google Books for: '{clean_title_text}' by '{clean_author_text}'")
        metadata_result = lookup_book_metadata_free(clean_title_text, clean_author_text)

        best_match = metadata_result.get("best_match", {})
        partial_matches = metadata_result.get("partial_matches", [])
        total_found = metadata_result.get("total_found", 0)

        if best_match:
            print(f"  ✓ Found {total_found} potential matches. Best match:")
            print(f"    Title: {best_match.get('title', 'Not found')}")
            print(f"    Authors: {', '.join(best_match.get('authors', []))}")
            print(f"    ISBN: {best_match.get('isbn', 'Not found')}")
            print(f"    Publisher: {best_match.get('publisher', 'Not found')}")
            print(f"    Published: {best_match.get('published_date', 'Not found')}")
            print(f"    Pages: {best_match.get('page_count', 'Not found')}")
            print(f"    Language: {best_match.get('language', 'Not found')}")
            print(f"    Match Strategy: {best_match.get('search_strategy', 'Unknown')}")
            print(f"    Confidence: {best_match.get('confidence_score', 0):.1f}%")

            if best_match.get("categories"):
                print(f"    Categories: {', '.join(best_match.get('categories', []))}")

            if best_match.get("description"):
                desc = (
                    best_match["description"][:150] + "..."
                    if len(best_match["description"]) > 150
                    else best_match["description"]
                )
                print(f"    Description: {desc}")

            # Show partial matches if available
            if partial_matches:
                print(f"  📚 Other possible matches ({len(partial_matches)}):")
                for idx, match in enumerate(partial_matches[:3], 1):  # Show top 3 alternatives
                    print(f"    {idx}. {match.get('title', 'Unknown')} by {', '.join(match.get('authors', ['Unknown']))}")
                    print(f"       Confidence: {match.get('confidence_score', 0):.1f}% | Strategy: {match.get('search_strategy', 'Unknown')}")
                    if match.get('isbn'):
                        print(f"       ISBN: {match.get('isbn')}")
                if len(partial_matches) > 3:
                    print(f"    ... and {len(partial_matches) - 3} more matches")
        else:
            print(f"  ✗ No Google Books results found")
    else:
        print(f"  ⚠ Title too short or unclear for API search")

    # Create enhanced result - convert numpy types to native Python types for JSON serialization
    enhanced_result = {
        "title": spine.title,
        "author": spine.author,
        "confidence": float(spine.confidence) if spine.confidence is not None else 0.0,
        "bbox": [int(x) for x in spine.bbox] if spine.bbox else [],
        "raw_text": spine.raw_text,
        "cleaned_title": clean_title_text,
        "cleaned_author": clean_author_text,
        "google_books_search": {
            "best_match": metadata_result.get("best_match", {}),
            "alternative_matches": metadata_result.get("partial_matches", []),
            "total_found": metadata_result.get("total_found", 0),
            "search_performed": bool(clean_title_text and len(clean_title_text) > 3)
        },
        # Keep backward compatibility
        "google_books_metadata": metadata_result.get("best_match", {}),
    }
    enhanced_results.append(enhanced_result)
    print()

print(f"\n=== SUMMARY ===")
print(f"OCR Provider Used: {detector.ocr_provider}")
print(f"Total book spines detected: {len(book_spines)}")
books_with_metadata = sum(
    1 for result in enhanced_results if result["google_books_search"]["best_match"]
)
total_matches_found = sum(
    result["google_books_search"]["total_found"] for result in enhanced_results
)
print(f"Books with Google Books matches: {books_with_metadata}")
print(f"Total potential matches found: {total_matches_found}")
success_rate = (
    (books_with_metadata / len(book_spines) * 100) if len(book_spines) > 0 else 0
)
print(f"Primary match success rate: {success_rate:.1f}%")

# Show distribution of match confidence
high_confidence_matches = sum(
    1 for result in enhanced_results
    if result["google_books_search"]["best_match"].get("confidence_score", 0) >= 70
)
if books_with_metadata > 0:
    avg_confidence = sum(
        result["google_books_search"]["best_match"].get("confidence_score", 0)
        for result in enhanced_results
        if result["google_books_search"]["best_match"]
    ) / books_with_metadata
    print(f"Average match confidence: {avg_confidence:.1f}%")
    print(f"High confidence matches (≥70%): {high_confidence_matches}")

# Add OCR accuracy info
if book_spines:
    avg_ocr_confidence = sum(spine.confidence for spine in book_spines) / len(
        book_spines
    )
    print(f"Average OCR confidence: {avg_ocr_confidence:.2f}")

# Export enhanced results
import json

with open("enhanced_library.json", "w", encoding="utf-8") as f:
    json.dump(enhanced_results, f, indent=2, ensure_ascii=False)

print(f"\nResults exported to 'enhanced_library.json'")

# Also export original format for compatibility
export_results(book_spines, "my_library.json")
