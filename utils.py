"""
Utilities Module

This module contains utility functions for:
- Loading and organizing the dataset
- Image loading and preprocessing
- Visualization
- Data splitting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm


def load_categories(categories_file):
    """
    Load category names from a file.
    
    Args:
        categories_file: Path to categories file
        
    Returns:
        List of category names
    """
    with open(categories_file, 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    return categories


def load_image_paths(label_file, base_dir):
    """
    Load image paths from a label file.
    
    Args:
        label_file: Path to label file
        base_dir: Base directory where images are stored
        
    Returns:
        List of image paths
    """
    with open(label_file, 'r') as f:
        image_paths = [os.path.join(base_dir, line.strip()) for line in f.readlines()]
    return image_paths


def load_dataset_split(label_file, base_dir, categories, target_size=(224, 224)):
    """
    Load images and labels for a dataset split.
    
    Args:
        label_file: Path to label file
        base_dir: Base directory where images are stored
        categories: List of category names
        target_size: Target size for images
        
    Returns:
        Tuple of (images, labels)
    """
    # Create a mapping from category name to index
    category_to_index = {cat: i for i, cat in enumerate(categories)}
    
    # Load image paths
    image_paths = load_image_paths(label_file, base_dir)
    
    # Initialize arrays for images and labels
    images = []
    labels = []
    
    print(f"Loading images from {label_file}...")
    for img_path in tqdm(image_paths):
        # Extract category from path
        parts = img_path.split('/')
        category = parts[-2]  # Assuming path is like base_dir/images/category/image.jpg
        
        if category in category_to_index:
            # Load and preprocess the image
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                images.append(img)
                labels.append(category_to_index[category])
    
    return np.array(images), np.array(labels)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to load image at {image_path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_minc_dataset(base_dir, fold=1, target_size=(224, 224)):
    """
    Load the MINC-2500 dataset for a specific fold.
    
    Args:
        base_dir: Base directory of the dataset
        fold: Fold number (1-5)
        target_size: Target size for images
        
    Returns:
        Dictionary containing train, validation, and test data
    """
    # Load categories
    categories_file = os.path.join(base_dir, 'categories.txt')
    categories = load_categories(categories_file)
    
    # Define label files
    train_label_file = os.path.join(base_dir, 'labels', f'train{fold}.txt')
    val_label_file = os.path.join(base_dir, 'labels', f'validate{fold}.txt')
    test_label_file = os.path.join(base_dir, 'labels', f'test{fold}.txt')
    
    # Load datasets
    X_train, y_train = load_dataset_split(train_label_file, base_dir, categories, target_size)
    X_val, y_val = load_dataset_split(val_label_file, base_dir, categories, target_size)
    X_test, y_test = load_dataset_split(test_label_file, base_dir, categories, target_size)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'categories': categories
    }


def create_train_val_test_split(images, labels, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Create train, validation, and test splits.
    
    Args:
        images: Images array
        labels: Labels array
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        Dictionary of split datasets
    """
    # First split into train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(val_size + test_size), stratify=labels, random_state=random_state
    )
    
    # Then split temp into validation and test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio), stratify=y_temp, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def create_tf_datasets(data_dict, batch_size=32, augment_train=True):
    """
    Create TensorFlow datasets from data dictionary.
    
    Args:
        data_dict: Dictionary with X_train, y_train, etc.
        batch_size: Batch size
        augment_train: Whether to apply data augmentation to training set
        
    Returns:
        Dictionary of TensorFlow datasets
    """
    from preprocessing import preprocess_batch_tf
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((data_dict['X_train'], data_dict['y_train']))
    val_dataset = tf.data.Dataset.from_tensor_slices((data_dict['X_val'], data_dict['y_val']))
    test_dataset = tf.data.Dataset.from_tensor_slices((data_dict['X_test'], data_dict['y_test']))
    
    # Prepare train dataset with shuffling and augmentation
    train_dataset = train_dataset.shuffle(buffer_size=len(data_dict['X_train']))
    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_batch_tf(tf.expand_dims(x, 0), augment=augment_train)[0], y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Prepare validation dataset
    val_dataset = val_dataset.map(
        lambda x, y: (preprocess_batch_tf(tf.expand_dims(x, 0), augment=False)[0], y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    # Prepare test dataset
    test_dataset = test_dataset.map(
        lambda x, y: (preprocess_batch_tf(tf.expand_dims(x, 0), augment=False)[0], y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }


def visualize_dataset_samples(images, labels, categories, n_samples=5, random_seed=None):
    """
    Visualize random samples from the dataset.
    
    Args:
        images: Images array
        labels: Labels array
        categories: List of category names
        n_samples: Number of samples to visualize
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Get random indices
    indices = random.sample(range(len(images)), min(n_samples, len(images)))
    
    # Create subplot grid
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        img = images[idx]
        label = labels[idx]
        category = categories[label]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {category}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_augmentations(image, n_augmentations=5):
    """
    Visualize different augmentations of an image.
    
    Args:
        image: Original image
        n_augmentations: Number of augmentations to visualize
    """
    from preprocessing import augment_image
    
    # Create subplot grid
    fig, axes = plt.subplots(1, n_augmentations+1, figsize=(15, 3))
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Display augmented images
    for i in range(n_augmentations):
        augmented = augment_image(image)
        axes[i+1].imshow(augmented)
        axes[i+1].set_title(f"Augmentation {i+1}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_segmentation(image, segmented_image, method_name=""):
    """
    Visualize original and segmented images side by side.
    
    Args:
        image: Original image
        segmented_image: Segmented image
        method_name: Name of the segmentation method
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    ax2.imshow(segmented_image, cmap='viridis' if len(segmented_image.shape) < 3 else None)
    ax2.set_title(f"Segmented Image ({method_name})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_feature_extraction(features, method_name, max_features=20):
    """
    Visualize extracted features.
    
    Args:
        features: Extracted features
        method_name: Name of the feature extraction method
        max_features: Maximum number of features to display
    """
    # Reshape if necessary
    if len(features.shape) > 1:
        # For feature vectors, take first sample
        features = features[0]
    
    # Limit number of features
    n_features = min(len(features), max_features)
    features = features[:n_features]
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(n_features), features)
    plt.title(f"Feature Visualization ({method_name})")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.show()


def plot_training_progress(history, metric='accuracy'):
    """
    Plot training progress during model training.
    
    Args:
        history: Training history
        metric: Metric to plot
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(f'{metric.capitalize()} During Training')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()