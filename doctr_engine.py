#!/usr/bin/env python3
import sys
import os
import json
import time
import warnings
import argparse
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Apply binary threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def run_doctr(image_path, preprocess=True):
    """
    Run DocTR on an image and return results in our standard format
    """
    try:
        # Initialize DocTR
        predictor = ocr_predictor(pretrained=True)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Preprocess if requested
        if preprocess:
            processed_image = preprocess_image(image)
        else:
            processed_image = image
        
        start_time = time.time()
        
        # Create DocumentFile from image bytes
        image_bytes = cv2.imencode('.png', processed_image)[1].tobytes()
        doc = DocumentFile.from_images([image_bytes])
        
        # Run OCR
        result = predictor(doc)
        
        processing_time = time.time() - start_time
        
        # Extract text and confidence scores
        extracted_texts = []
        confidences = []
        
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.value and word.value.strip():
                            extracted_texts.append(word.value.strip())
                            confidences.append(word.confidence)
        
        # Join all extracted texts
        full_text = " ".join(extracted_texts)
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Calculate word count
        num_words = len(full_text.split()) if full_text else 0
        
        return {
            "engine": "doctr",
            "image": image_path,
            "text": full_text,
            "avg_confidence": avg_confidence,
            "processing_time": processing_time,
            "num_words": num_words
        }
        
    except Exception as e:
        print(f"Error running DocTR: {e}", file=sys.stderr)
        return {
            "engine": "doctr",
            "image": image_path,
            "text": "",
            "avg_confidence": 0.0,
            "processing_time": 0.0,
            "num_words": 0,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Run DocTR on an image')
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--no-preprocess', action='store_true', help='Skip image preprocessing')
    parser.add_argument('--out', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    
    # Run DocTR
    result = run_doctr(args.image, preprocess=not args.no_preprocess)
    
    # Output results
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.out}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()


