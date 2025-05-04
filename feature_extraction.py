"""
Feature Extraction Module

This module contains functions for extracting features from images:
- MobileNetV2 feature extraction
- Custom feature extraction methods
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
import tensorflow as tf


def create_mobilenetv2_extractor(input_shape=(224, 224, 3), include_top=False):
    """
    Create a MobileNetV2 feature extractor.
    
    Args:
        input_shape: Shape of input images
        include_top: Whether to include the fully-connected layer at the top
        
    Returns:
        MobileNetV2 model for feature extraction
    """
    # Create a simple feature extractor using MobileNetV2 in a way that's more compatible
    try:
        # Try the standard approach first
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=include_top,
            weights='imagenet'
        )
        
        # Create our own model with global average pooling
        inputs = Input(shape=input_shape)
        x = base_model(inputs)
        outputs = GlobalAveragePooling2D()(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    except Exception as e:
        print(f"Error creating MobileNetV2 feature extractor: {e}")
        print("Falling back to a simpler feature extractor...")
        
        # Create a simpler CNN feature extractor
        model = tf.keras.Sequential([
            Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        return model


def extract_mobilenetv2_features(images, model=None):
    """
    Extract features using MobileNetV2.
    
    Args:
        images: Input images (already preprocessed)
        model: MobileNetV2 model instance
        
    Returns:
        Feature vectors for the input images
    """
    if model is None:
        model = create_mobilenetv2_extractor()
    
    # Ensure images are in correct format
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)  # Add batch dimension for single image
    
    # Make predictions in smaller batches to avoid memory issues
    try:
        features = model.predict(images, batch_size=16)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        print("Trying with CPU...")
        
        with tf.device('/cpu:0'):
            features = model.predict(images, batch_size=8)
    
    return features


def extract_color_histogram(image, bins=32, channels=(0, 1, 2), mask=None):
    """
    Extract color histogram features.
    
    Args:
        image: Input image
        bins: Number of bins per channel
        channels: Color channels to use
        mask: Optional mask for histogram calculation
        
    Returns:
        Histogram features
    """
    hist = []
    
    for i, channel in enumerate(channels):
        if len(image.shape) > 2 and i < image.shape[2]:
            channel_hist = cv2.calcHist([image], [channel], mask, [bins], [0, 256])
            channel_hist = cv2.normalize(channel_hist, channel_hist).flatten()
            hist.extend(channel_hist)
    
    return np.array(hist)


def extract_texture_features_glcm(image, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract texture features using Gray-Level Co-occurrence Matrix (GLCM).
    
    Args:
        image: Input grayscale image
        distances: List of pixel pair distances
        angles: List of pixel pair angles
        
    Returns:
        GLCM features
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to 8 levels to reduce computation
    levels = 8
    gray = (gray / (256 / levels)).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Combine all features
    features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
    
    return features


def extract_lbp_features(image, n_points=24, radius=3, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features.
    
    Args:
        image: Input grayscale image
        n_points: Number of circularly symmetric neighbor points
        radius: Radius of circle
        method: Method to determine the LBP
        
    Returns:
        LBP histogram features
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method)
    
    # Calculate histogram
    n_bins = n_points + 2 if method == 'uniform' else 2**n_points
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Args:
        image: Input image
        orientations: Number of orientation bins
        pixels_per_cell: Size (in pixels) of a cell
        cells_per_block: Size (in cells) of a block
        
    Returns:
        HOG features
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to reduce computation
    resized = cv2.resize(gray, (128, 128))
    
    # Extract HOG features
    features = hog(
        resized, 
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        block_norm='L2-Hys', 
        visualize=False
    )
    
    return features


def extract_shape_features(image):
    """
    Extract shape features (moments, area, perimeter).
    
    Args:
        image: Binary image (thresholded)
        
    Returns:
        Shape features
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold if not already binary
    if np.max(gray) > 1:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = gray
    
    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate contour features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate Hu Moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments)
        
        # Add features
        features = [area, perimeter, aspect_ratio]
        features.extend(hu_moments.flatten())
    else:
        # No contours found, return zeros
        features = np.zeros(10)  # 3 simple features + 7 Hu moments
    
    return np.array(features)


def extract_combined_features(image, feature_types=None):
    """
    Extract multiple types of features and combine them.
    
    Args:
        image: Input image
        feature_types: List of feature types to extract
        
    Returns:
        Combined feature vector
    """
    if feature_types is None:
        feature_types = ['color', 'texture', 'lbp', 'hog']
    
    features = []
    
    if 'color' in feature_types:
        color_hist = extract_color_histogram(image)
        features.append(color_hist)
    
    if 'texture' in feature_types:
        texture_features = extract_texture_features_glcm(image)
        features.append(texture_features)
    
    if 'lbp' in feature_types:
        lbp_features = extract_lbp_features(image)
        features.append(lbp_features)
    
    if 'hog' in feature_types:
        hog_features = extract_hog_features(image)
        features.append(hog_features)
    
    if 'shape' in feature_types:
        # Create a binary image for shape feature extraction
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        shape_features = extract_shape_features(binary)
        features.append(shape_features)
    
    # Concatenate all features
    if features:
        return np.concatenate(features)
    else:
        return np.array([])


def batch_extract_features(images, method='mobilenetv2', **kwargs):
    """
    Extract features from a batch of images using the specified method.
    
    Args:
        images: Batch of input images
        method: Feature extraction method
        kwargs: Additional arguments for the extraction method
        
    Returns:
        Feature vectors for all images
    """
    if method == 'mobilenetv2':
        model = create_mobilenetv2_extractor()
        return extract_mobilenetv2_features(images, model)
    elif method == 'color':
        return np.array([extract_color_histogram(img, **kwargs) for img in images])
    elif method == 'texture':
        return np.array([extract_texture_features_glcm(img, **kwargs) for img in images])
    elif method == 'lbp':
        return np.array([extract_lbp_features(img, **kwargs) for img in images])
    elif method == 'hog':
        return np.array([extract_hog_features(img, **kwargs) for img in images])
    elif method == 'combined':
        return np.array([extract_combined_features(img, **kwargs) for img in images])
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")