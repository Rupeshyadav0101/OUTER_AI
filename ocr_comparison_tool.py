#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add engines directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

# Import OCR engines
from engines.easyocr_engine import run_easyocr
from engines.tesseract_engine import run_tesseract
from engines.doctr_engine import run_doctr
from engines.paddleocr_engine import run_paddleocr

# Import evaluation functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def preprocess_text(text):
    """Clean and normalize text for comparison"""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Preprocess texts
    text1_clean = preprocess_text(text1)
    text2_clean = preprocess_text(text2)
    
    if not text1_clean or not text2_clean:
        return 0.0
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0

def run_ocr_comparison(image_path, ground_truth=""):
    """Run all OCR engines and return comparison results"""
    results = {}
    
    # List of OCR engines to test
    engines = [
        ("EasyOCR", run_easyocr),
        ("Tesseract", run_tesseract),
        ("DocTR", run_doctr),
        ("PaddleOCR", run_paddleocr)
    ]
    
    for engine_name, engine_func in engines:
        try:
            st.info(f"Running {engine_name}...")
            
            # Run OCR
            start_time = time.time()
            result = engine_func(image_path)
            processing_time = time.time() - start_time
            
            if result and "text" in result:
                extracted_text = result.get("text", "")
                confidence = result.get("avg_confidence", 0.0)
                word_count = result.get("num_words", 0)
                
                # Calculate accuracy if ground truth provided
                accuracy = 0.0
                if ground_truth:
                    accuracy = calculate_cosine_similarity(extracted_text, ground_truth)
                
                results[engine_name] = {
                    "extracted_text": extracted_text,
                    "processing_time": processing_time,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "word_count": word_count,
                    "status": "success"
                }
            else:
                results[engine_name] = {
                    "extracted_text": "",
                    "processing_time": processing_time,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "word_count": 0,
                    "status": "failed",
                    "error": "No text extracted"
                }
                
        except Exception as e:
            results[engine_name] = {
                "extracted_text": "",
                "processing_time": 0.0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "word_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    return results

def create_comparison_charts(results):
    """Create comparison charts"""
    # Prepare data for charts
    engines = list(results.keys())
    processing_times = [results[engine]["processing_time"] for engine in engines]
    accuracies = [results[engine]["accuracy"] * 100 for engine in engines]  # Convert to percentage
    confidences = [results[engine]["confidence"] * 100 for engine in engines]  # Convert to percentage
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Processing Time Comparison
    bars1 = ax1.bar(engines, processing_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}s', ha='center', va='bottom')
    
    # 2. Accuracy Comparison
    bars2 = ax2.bar(engines, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('Accuracy Comparison (Cosine Similarity)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. Confidence Comparison
    bars3 = ax3.bar(engines, confidences, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('Confidence Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Confidence (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 4. Speed vs Accuracy Scatter Plot
    ax4.scatter(processing_times, accuracies, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_xlabel('Processing Time (seconds)')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Speed vs Accuracy', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add engine labels to scatter plot
    for i, engine in enumerate(engines):
        ax4.annotate(engine, (processing_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(
        page_title="OCR Comparison Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 OCR Engine Comparison Tool")
    st.markdown("Compare the performance of 4 different OCR engines: EasyOCR, Tesseract, DocTR, and PaddleOCR")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image to analyze",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing text to compare OCR engines"
    )
    
    # Ground truth input
    ground_truth = st.sidebar.text_area(
        "Ground Truth Text (Optional)",
        height=150,
        help="Enter the expected text from the image to calculate accuracy"
    )
    
    # Run comparison button
    run_comparison = st.sidebar.button("🚀 Run OCR Comparison", type="primary")
    
    # Main content area
    if uploaded_file is not None:
        # Display uploaded image
        st.header("📸 Uploaded Image")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.info("**Image Details:**")
            st.write(f"**File Name:** {uploaded_file.name}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**File Type:** {uploaded_file.type}")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if run_comparison:
            st.header("🔄 Running OCR Comparison")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run comparison
            results = run_ocr_comparison(temp_path, ground_truth)
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("✅ Comparison completed!")
            
            # Display results
            st.header("📊 Comparison Results")
            
            # Create DataFrame for results
            results_data = []
            for engine, result in results.items():
                results_data.append({
                    "Engine": engine,
                    "Status": result["status"],
                    "Processing Time (s)": f"{result['processing_time']:.3f}",
                    "Accuracy (%)": f"{result['accuracy'] * 100:.1f}",
                    "Confidence (%)": f"{result['confidence'] * 100:.1f}",
                    "Word Count": result["word_count"]
                })
            
            # Display results table
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
            
            # Create and display charts
            st.header("📈 Performance Charts")
            try:
                fig = create_comparison_charts(results)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating charts: {e}")
            
            # Display extracted text for each engine
            st.header("📝 Extracted Text Comparison")
            
            # Create tabs for each engine
            tabs = st.tabs([engine for engine in results.keys()])
            
            for i, (engine, result) in enumerate(results.items()):
                with tabs[i]:
                    if result["status"] == "success":
                        st.success(f"✅ {engine} completed successfully")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Processing Time", f"{result['processing_time']:.3f}s")
                            st.metric("Word Count", result["word_count"])
                        
                        with col2:
                            st.metric("Accuracy", f"{result['accuracy'] * 100:.1f}%")
                            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                        
                        st.subheader("Extracted Text:")
                        st.text_area(
                            f"{engine} Results",
                            value=result["extracted_text"],
                            height=200,
                            key=f"text_{engine}"
                        )
                        
                    elif result["status"] == "failed":
                        st.error(f"❌ {engine} failed")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
                    else:
                        st.warning(f"⚠️ {engine} encountered an error")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
            
            # Summary
            st.header("🏆 Summary")
            
            # Find best performing engine
            successful_results = {k: v for k, v in results.items() if v["status"] == "success"}
            
            if successful_results:
                # Best by accuracy
                best_accuracy = max(successful_results.items(), key=lambda x: x[1]["accuracy"])
                
                # Best by speed
                best_speed = min(successful_results.items(), key=lambda x: x[1]["processing_time"])
                
                # Best by confidence
                best_confidence = max(successful_results.items(), key=lambda x: x[1]["confidence"])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Best Accuracy",
                        best_accuracy[0],
                        f"{best_accuracy[1]['accuracy'] * 100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "⚡ Fastest",
                        best_speed[0],
                        f"{best_speed[1]['processing_time']:.3f}s"
                    )
                
                with col3:
                    st.metric(
                        "🎯 Most Confident",
                        best_confidence[0],
                        f"{best_confidence[1]['confidence'] * 100:.1f}%"
                    )
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Show sample usage
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload an image** containing text using the file uploader in the sidebar
        2. **Optionally enter ground truth text** to calculate accuracy
        3. **Click 'Run OCR Comparison'** to analyze the image with all 4 OCR engines
        4. **View the results** including:
           - Processing time comparison
           - Accuracy analysis (if ground truth provided)
           - Confidence scores
           - Extracted text from each engine
           - Performance charts and graphs
        """)
        
        st.header("🔧 Supported OCR Engines")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **EasyOCR**: Deep learning-based OCR
            - **Tesseract**: Google's open-source OCR
            """)
        
        with col2:
            st.markdown("""
            - **DocTR**: Document Text Recognition
            - **PaddleOCR**: Baidu's OCR framework
            """)

if __name__ == "__main__":
    main()
