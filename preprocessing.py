"""
Preprocessing and Image Enhancement Module

This module contains functions for image preprocessing and enhancement:
- Image Resizing
- Normalization
- Noise Reduction
- Contrast Adjustment
- Color Space Conversion
- Image Augmentation
- Thresholding
- Blurring and Sharpening Filters
- Morphological Operations
"""

import numpy as np
import cv2
from skimage import exposure
import tensorflow as tf


def resize_image(image, target_size=(224, 224)):
    """Resize an image to target size."""
    return cv2.resize(image, target_size)


def normalize_image(image, method='minmax'):
    """
    Normalize image values.
    
    Args:
        image: Input image
        method: 'minmax' (0-1) or 'standardize' (mean=0, std=1)
    
    Returns:
        Normalized image
    """
    if method == 'minmax':
        # Scale to [0, 1]
        return image.astype(np.float32) / 255.0
    elif method == 'standardize':
        # Standardize to mean=0, std=1
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def reduce_noise(image, method='gaussian', kernel_size=5):
    """
    Apply noise reduction to image.
    
    Args:
        image: Input image
        method: 'gaussian', 'median', or 'bilateral'
        kernel_size: Size of the kernel for filtering
        
    Returns:
        Filtered image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError(f"Unknown noise reduction method: {method}")


def adjust_contrast(image, method='histogram_equalization'):
    """
    Adjust image contrast.
    
    Args:
        image: Input image
        method: 'histogram_equalization', 'clahe', or 'gamma'
        
    Returns:
        Contrast-adjusted image
    """
    if method == 'histogram_equalization':
        # For color images, apply to value channel in HSV
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return cv2.equalizeHist(image)
    elif method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = clahe.apply(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return clahe.apply(image)
    elif method == 'gamma':
        # Gamma correction (gamma = 1.5)
        return exposure.adjust_gamma(image, 1.5)
    else:
        raise ValueError(f"Unknown contrast adjustment method: {method}")


def convert_color_space(image, target_space='hsv'):
    """
    Convert image to different color space.
    
    Args:
        image: Input image (BGR format)
        target_space: 'hsv', 'lab', or 'grayscale'
        
    Returns:
        Converted image
    """
    if target_space == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif target_space == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif target_space == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown color space: {target_space}")


def apply_thresholding(image, method='otsu'):
    """
    Apply thresholding to an image.
    
    Args:
        image: Input image (grayscale)
        method: 'binary', 'otsu', or 'adaptive'
        
    Returns:
        Thresholded binary image
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    if method == 'binary':
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    else:
        raise ValueError(f"Unknown thresholding method: {method}")
        
    return thresh


def apply_filter(image, filter_type='blur', kernel_size=5):
    """
    Apply blurring or sharpening filter.
    
    Args:
        image: Input image
        filter_type: 'blur', 'gaussian', 'median', or 'sharpen'
        kernel_size: Size of kernel for filtering
        
    Returns:
        Filtered image
    """
    if filter_type == 'blur':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif filter_type == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == 'sharpen':
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def apply_morphology(image, operation='erosion', kernel_size=5):
    """
    Apply morphological operations.
    
    Args:
        image: Binary input image
        operation: 'erosion', 'dilation', 'opening', or 'closing'
        kernel_size: Size of structuring element
        
    Returns:
        Processed image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unknown morphological operation: {operation}")


def augment_image(image, augmentations=None):
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image
        augmentations: List of augmentations to apply
        
    Returns:
        Augmented image
    """
    if augmentations is None:
        augmentations = ['flip', 'rotate', 'brightness']
    
    img = image.copy()
    
    if 'flip' in augmentations:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
    
    if 'rotate' in augmentations:
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
    
    if 'brightness' in augmentations:
        # Random brightness adjustment
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = hsv[:,:,2] * np.random.uniform(0.8, 1.2)
            hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img


def preprocess_pipeline(image, target_size=(224, 224), normalize=True, 
                         reduce_noise_method=None, enhance_contrast=None,
                         augment=False):
    """
    Complete preprocessing pipeline.
    
    Args:
        image: Input image
        target_size: Size to resize image to
        normalize: Whether to normalize pixel values
        reduce_noise_method: Noise reduction method (None for no noise reduction)
        enhance_contrast: Contrast enhancement method (None for no enhancement)
        augment: Whether to apply data augmentation
        
    Returns:
        Preprocessed image
    """
    # Resize image
    img = resize_image(image, target_size)
    
    # Apply data augmentation if requested
    if augment:
        img = augment_image(img)
    
    # Apply noise reduction if specified
    if reduce_noise_method:
        img = reduce_noise(img, method=reduce_noise_method)
    
    # Apply contrast enhancement if specified
    if enhance_contrast:
        img = adjust_contrast(img, method=enhance_contrast)
    
    # Normalize pixel values
    if normalize:
        img = normalize_image(img)
    
    return img


def preprocess_batch_tf(images, target_size=(224, 224), augment=False):
    """
    Preprocess a batch of images using TensorFlow operations.
    
    Args:
        images: Batch of input images
        target_size: Size to resize images to
        augment: Whether to apply data augmentation
        
    Returns:
        Preprocessed batch of images
    """
    # Convert to float32 and scale to [0, 1]
    images = tf.cast(images, tf.float32) / 255.0
    
    # Resize images
    images = tf.image.resize(images, target_size)
    
    if augment:
        # Random flip
        images = tf.image.random_flip_left_right(images)
        
        # Random brightness
        images = tf.image.random_brightness(images, 0.2)
        
        # Random contrast
        images = tf.image.random_contrast(images, 0.8, 1.2)
        
        # Random hue
        images = tf.image.random_hue(images, 0.1)
        
        # Random saturation
        images = tf.image.random_saturation(images, 0.8, 1.2)
    
    # Clip values to [0, 1]
    images = tf.clip_by_value(images, 0.0, 1.0)
    
    # Normalize for MobileNetV2
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images * 255.0)
    
    return images