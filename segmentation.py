"""
Segmentation Module

This module contains functions for image segmentation:
- Thresholding techniques
- Edge-based methods
- Region-based methods
- Clustering approaches
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import ndimage


def simple_thresholding(image, threshold=127):
    """
    Apply simple thresholding to segment the image.
    
    Args:
        image: Input image (grayscale)
        threshold: Threshold value
        
    Returns:
        Binary segmented image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def otsu_thresholding(image):
    """
    Apply Otsu's thresholding to segment the image.
    
    Args:
        image: Input image (grayscale)
        
    Returns:
        Binary segmented image and threshold value
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, threshold


def adaptive_thresholding(image, block_size=11, C=2):
    """
    Apply adaptive thresholding to segment the image.
    
    Args:
        image: Input image (grayscale)
        block_size: Size of the block for adaptive thresholding
        C: Constant subtracted from the mean
        
    Returns:
        Binary segmented image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, block_size, C)
    return binary


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to segment the image.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for the hysteresis procedure
        high_threshold: Higher threshold for the hysteresis procedure
        
    Returns:
        Edge image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to segment the image.
    
    Args:
        image: Input image
        
    Returns:
        Edge image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Sobel operators
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to range [0, 255]
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    
    return magnitude


def region_growing(image, seed_point, threshold=10):
    """
    Apply region growing segmentation.
    
    Args:
        image: Input image (grayscale)
        seed_point: (x, y) coordinates of the seed point
        threshold: Intensity difference threshold for including pixels
        
    Returns:
        Binary mask of the segmented region
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create a mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    
    # Get the seed intensity
    seed_intensity = gray[seed_point[1], seed_point[0]]
    
    # Create a queue for breadth-first search
    queue = [seed_point]
    visited = set([(seed_point[0], seed_point[1])])
    
    height, width = gray.shape
    
    # 4-connectivity neighborhood
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        x, y = queue.pop(0)
        mask[y, x] = 255
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            # Check if the neighbor is inside the image
            if 0 <= nx < width and 0 <= ny < height:
                # Check if the neighbor has not been visited and has a similar intensity
                if (nx, ny) not in visited and abs(int(gray[ny, nx]) - int(seed_intensity)) <= threshold:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
    
    return mask


def watershed_segmentation(image):
    """
    Apply watershed segmentation.
    
    Args:
        image: Input color image
        
    Returns:
        Segmented image with markers
    """
    if len(image.shape) == 2:
        # Convert grayscale to color
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundary in red
    
    return markers, image


def kmeans_segmentation(image, k=3):
    """
    Apply K-means clustering for segmentation.
    
    Args:
        image: Input image
        k: Number of clusters
        
    Returns:
        Segmented image
    """
    # Reshape image for clustering
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to 8-bit values
    centers = np.uint8(centers)
    
    # Map labels to center values
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image, labels.reshape(image.shape[:2])


def mean_shift_segmentation(image, spatial_radius=10, color_radius=40, min_density=50):
    """
    Apply mean shift segmentation.
    
    Args:
        image: Input image
        spatial_radius: Spatial window radius
        color_radius: Color window radius
        min_density: Minimum point density
        
    Returns:
        Segmented image
    """
    # Mean shift segmentation using OpenCV's pyrMeanShiftFiltering
    segmented = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius, min_density)
    return segmented


def grabcut_segmentation(image, rect=None):
    """
    Apply GrabCut segmentation.
    
    Args:
        image: Input image
        rect: Rectangle containing the object in the format (x, y, width, height)
        
    Returns:
        Binary mask and segmented image
    """
    # Create mask and initialize with obvious background (0) and foreground (1)
    mask = np.zeros(image.shape[:2], np.uint8)
    
    if rect is None:
        # Use a default rectangle that covers most of the image
        h, w = image.shape[:2]
        margin = int(min(h, w) * 0.1)
        rect = (margin, margin, w - 2*margin, h - 2*margin)
    
    # Background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify mask: 0 and 2 (background) -> 0, 1 and 3 (foreground) -> 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Create segmented image
    segmented = image * mask2[:, :, np.newaxis]
    
    return mask2, segmented


def segment_image(image, method='otsu', **kwargs):
    """
    Segment an image using the specified method.
    
    Args:
        image: Input image
        method: Segmentation method to use
        kwargs: Additional arguments specific to the chosen method
        
    Returns:
        Segmented image
    """
    if method == 'simple_threshold':
        return simple_thresholding(image, **kwargs)
    elif method == 'otsu':
        return otsu_thresholding(image)[0]
    elif method == 'adaptive':
        return adaptive_thresholding(image, **kwargs)
    elif method == 'canny':
        return canny_edge_detection(image, **kwargs)
    elif method == 'sobel':
        return sobel_edge_detection(image)
    elif method == 'kmeans':
        return kmeans_segmentation(image, **kwargs)[0]
    elif method == 'watershed':
        return watershed_segmentation(image)[1]
    elif method == 'mean_shift':
        return mean_shift_segmentation(image, **kwargs)
    elif method == 'grabcut':
        return grabcut_segmentation(image, **kwargs)[1]
    else:
        raise ValueError(f"Unknown segmentation method: {method}")