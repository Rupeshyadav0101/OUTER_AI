#!/usr/bin/env python3
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from paddleocr import PaddleOCR

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        return
    
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Initialize OCR
    ocr = PaddleOCR(lang="en", use_textline_orientation=True)
    
    # Extract text
    result = ocr.predict(image_path)
    
    if not result:
        print("No text detected.")
        return
    
    print("EXTRACTED TEXT:")
    print("=" * 40)
    
    # Extract text from the result structure
    for res in result:
        # Check if it's the new PaddleOCR structure
        if hasattr(res, 'res') and isinstance(res.res, dict):
            if 'rec_texts' in res.res:
                texts = res.res['rec_texts']
                for text in texts:
                    if text and text.strip():
                        print(text.strip())
        # Check if it has ocr_text attribute
        elif hasattr(res, 'ocr_text'):
            if res.ocr_text and res.ocr_text.strip():
                print(res.ocr_text.strip())
        # Fallback: try to access as dict
        else:
            try:
                if isinstance(res, dict) and 'rec_texts' in res:
                    for text in res['rec_texts']:
                        if text and text.strip():
                            print(text.strip())
            except:
                pass

if __name__ == "__main__":
    main()