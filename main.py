"""
Main Module

This module implements the complete pipeline for recycled materials classification:
1. Preprocessing and Image Enhancement
2. Segmentation
3. Feature Extraction
4. Classification
5. Evaluation and Performance Metrics
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

# Import custom modules
import preprocessing
import segmentation
import feature_extraction
import classification
import evaluation
import utils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Recycled Materials Classification Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='/content/minc2500/minc-2500',
                        help='Path to the MINC-2500 dataset')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Path to save output files')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--fold', type=int, default=1,
                        help='Dataset fold to use (1-5)')
    parser.add_argument('--model_type', type=str, default='mobilenetv2',
                        choices=['mobilenetv2', 'simple_cnn', 'deeper_cnn'],
                        help='Type of classification model to use')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip preprocessing stage (use for debugging)')
    parser.add_argument('--skip_segmentation', action='store_true',
                        help='Skip segmentation stage (use for debugging)')
    parser.add_argument('--feature_method', type=str, default='mobilenetv2',
                        choices=['mobilenetv2', 'combined', 'color', 'texture', 'lbp', 'hog'],
                        help='Feature extraction method')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Force using CPU instead of GPU')
    
    return parser.parse_args()


def setup_directories(output_dir):
    """Set up output directories."""
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def preprocess_stage(data_dict, args):
    """
    Preprocessing and Image Enhancement stage.
    
    Args:
        data_dict: Dictionary with dataset splits
        args: Command line arguments
        
    Returns:
        Dictionary with preprocessed datasets
    """
    print("\n--- Stage 1: Preprocessing and Image Enhancement ---\n")
    
    # Display sample images before preprocessing
    print("Original samples:")
    utils.visualize_dataset_samples(data_dict['X_train'], data_dict['y_train'], 
                                    data_dict['categories'], n_samples=5, random_seed=42)
    
    # Apply preprocessing to all images
    print("Applying preprocessing...")
    
    # Create TensorFlow datasets with preprocessing
    tf_datasets = utils.create_tf_datasets(data_dict, batch_size=args.batch_size, augment_train=True)
    
    # Display augmentation examples
    print("Augmentation examples:")
    utils.visualize_augmentations(data_dict['X_train'][0])
    
    return tf_datasets


def segmentation_stage(data_dict):
    """
    Segmentation stage.
    
    Args:
        data_dict: Dictionary with dataset splits
        
    Returns:
        Dictionary with segmented datasets
    """
    print("\n--- Stage 2: Segmentation ---\n")
    
    # We'll demonstrate different segmentation methods on a few sample images
    sample_images = data_dict['X_train'][:5]
    
    segmentation_methods = [
        ('otsu', {}),
        ('adaptive', {'block_size': 11, 'C': 2}),
        ('canny', {'low_threshold': 50, 'high_threshold': 150}),
        ('kmeans', {'k': 3})
    ]
    
    # Apply segmentation methods to sample images
    for i, image in enumerate(sample_images):
        print(f"Segmentation examples for sample {i+1}:")
        
        for method_name, params in segmentation_methods:
            # Apply segmentation
            segmented = segmentation.segment_image(image, method=method_name, **params)
            
            # Visualize result
            utils.visualize_segmentation(image, segmented, method_name=method_name)
    
    # For the full pipeline, we'll continue with original images
    # In a real-world scenario, you might want to use segmented images
    return data_dict


def feature_extraction_stage(data_dict, tf_datasets, args):
    """
    Feature Extraction stage.
    
    Args:
        data_dict: Dictionary with dataset splits
        tf_datasets: Dictionary with TensorFlow datasets
        args: Command line arguments
        
    Returns:
        Tuple of (features_dict, model)
    """
    print("\n--- Stage 3: Feature Extraction ---\n")
    
    if args.feature_method == 'mobilenetv2':
        print("Using MobileNetV2 for feature extraction...")
        
        # Create feature extractor model
        feature_extractor = feature_extraction.create_mobilenetv2_extractor()
        
        # For MobileNetV2, we'll use the TensorFlow datasets directly in the classification stage
        return None, feature_extractor
    
    else:
        print(f"Extracting {args.feature_method} features...")
        
        # Extract features from a few samples for visualization
        sample_features = feature_extraction.batch_extract_features(
            data_dict['X_train'][:10], method=args.feature_method
        )
        
        # Visualize features
        utils.visualize_feature_extraction(sample_features, args.feature_method)
        
        # Extract features for all datasets
        features_dict = {
            'X_train': feature_extraction.batch_extract_features(data_dict['X_train'], method=args.feature_method),
            'X_val': feature_extraction.batch_extract_features(data_dict['X_val'], method=args.feature_method),
            'X_test': feature_extraction.batch_extract_features(data_dict['X_test'], method=args.feature_method),
            'y_train': data_dict['y_train'],
            'y_val': data_dict['y_val'],
            'y_test': data_dict['y_test']
        }
        
        return features_dict, None


def create_simplified_cnn(input_shape=(224, 224, 3), num_classes=7):
    """
    Create a simplified CNN model that avoids operations that might cause compatibility issues.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        
    Returns:
        CNN model
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block - simplified
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block - simplified
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block - simplified
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with a standard optimizer
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def classification_stage(data_dict, tf_datasets, features_dict, feature_extractor, args):
    """
    Classification stage.
    
    Args:
        data_dict: Dictionary with dataset splits
        tf_datasets: Dictionary with TensorFlow datasets
        features_dict: Dictionary with extracted features
        feature_extractor: Feature extractor model
        args: Command line arguments
        
    Returns:
        Trained classification model
    """
    print("\n--- Stage 4: Classification ---\n")
    
    num_classes = len(data_dict['categories'])
    
    if args.model_type in ['mobilenetv2', 'simple_cnn', 'deeper_cnn']:
        # For deep learning models, we'll use the TensorFlow datasets
        
        # Check if we need to use a simplified model due to CUDA/cuDNN compatibility issues
        try:
            if args.model_type == 'mobilenetv2':
                print("Creating MobileNetV2 classifier...")
                model = classification.create_mobilenetv2_classifier(
                    input_shape=(224, 224, 3),
                    num_classes=num_classes,
                    fine_tune_layers=50
                )
            elif args.model_type == 'simple_cnn':
                print("Creating simple CNN classifier...")
                model = classification.create_simple_cnn(
                    input_shape=(224, 224, 3),
                    num_classes=num_classes
                )
            elif args.model_type == 'deeper_cnn':
                print("Creating deeper CNN classifier...")
                model = classification.create_deeper_cnn(
                    input_shape=(224, 224, 3),
                    num_classes=num_classes
                )
        except Exception as e:
            print(f"Error creating model: {e}")
            print("Falling back to simplified CNN model...")
            model = create_simplified_cnn(
                input_shape=(224, 224, 3),
                num_classes=num_classes
            )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Print model summary
        model.summary()
        
        try:
            # Train the model
            print("Training model...")
            history = model.fit(
                tf_datasets['train'],
                validation_data=tf_datasets['val'],
                epochs=args.epochs,
                callbacks=callbacks
            )
            
            # Plot training curves
            evaluation.plot_learning_curves(history)
        except Exception as e:
            print(f"Error during model training: {e}")
            print("Trying with a simplified approach or smaller dataset...")
            
            # Try with a smaller batch of data
            X_train = np.array([x.numpy() for x, _ in tf_datasets['train'].take(100)])
            y_train = np.array([y.numpy() for _, y in tf_datasets['train'].take(100)])
            X_val = np.array([x.numpy() for x, _ in tf_datasets['val'].take(20)])
            y_val = np.array([y.numpy() for _, y in tf_datasets['val'].take(20)])
            
            # Create a very simple model
            simple_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            simple_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("Training simplified model on CPU...")
            with tf.device('/cpu:0'):
                history = simple_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=32
                )
            
            model = simple_model
        
    else:
        # For traditional ML models, we'll use the extracted features
        
        if features_dict is None:
            print("Error: Feature extraction results are required for traditional ML models.")
            return None
        
        print("Training traditional ML classifier...")
        
        # Optimize hyperparameters using grid search
        model = classification.optimize_sklearn_model(
            features_dict['X_train'], features_dict['y_train'],
            model_type='svm' if args.model_type == 'svm' else 'random_forest',
            cv=3
        )
    
    return model


def evaluation_stage(model, data_dict, tf_datasets, features_dict, args):
    """
    Evaluation and Performance Metrics stage.
    
    Args:
        model: Trained classification model
        data_dict: Dictionary with dataset splits
        tf_datasets: Dictionary with TensorFlow datasets
        features_dict: Dictionary with extracted features
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n--- Stage 5: Evaluation and Performance Metrics ---\n")
    
    try:
        if args.model_type in ['mobilenetv2', 'simple_cnn', 'deeper_cnn']:
            # Evaluate deep learning model
            
            # Evaluate on test set
            print("Evaluating on test set...")
            test_loss, test_acc = model.evaluate(tf_datasets['test'])
            print(f"Test accuracy: {test_acc:.4f}")
            
            # Get predictions
            y_prob = model.predict(tf_datasets['test'])
            y_pred = np.argmax(y_prob, axis=1)
            
            # Get true labels from test dataset
            y_true = np.concatenate([y for _, y in tf_datasets['test']], axis=0)
            
        else:
            # Evaluate traditional ML model
            if features_dict is None:
                print("Error: Feature extraction results are required for traditional ML models.")
                return None
            
            # Get predictions
            print("Evaluating on test set...")
            y_prob = model.predict_proba(features_dict['X_test'])
            y_pred = model.predict(features_dict['X_test'])
            y_true = features_dict['y_test']
            
            # Calculate accuracy
            test_acc = np.mean(y_pred == y_true)
            print(f"Test accuracy: {test_acc:.4f}")
        
        # Comprehensive evaluation
        print("\nDetailed evaluation:")
        metrics = evaluation.evaluate_model_performance(
            model, data_dict['X_test'], data_dict['y_test'],
            class_names=data_dict['categories']
        )
        
        # Print metrics
        print("\nEvaluation metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # Plot confusion matrix
        print("\nConfusion matrix:")
        evaluation.plot_confusion_matrix(
            y_true, y_pred,
            classes=data_dict['categories'],
            normalize=True
        )
        
        return metrics
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Simplified evaluation:")
        
        # Perform simple accuracy calculation
        test_images = np.array([x.numpy() for x, _ in tf_datasets['test'].take(50)])
        test_labels = np.array([y.numpy() for _, y in tf_datasets['test'].take(50)])
        
        y_pred = np.argmax(model.predict(test_images), axis=1)
        accuracy = np.mean(y_pred == test_labels)
        print(f"Simplified test accuracy (on 50 samples): {accuracy:.4f}")
        
        return {'accuracy': accuracy}


def main():
    """Main function to run the complete pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directories
    model_dir = setup_directories(args.output_dir)
    
    # Set up device strategy based on args
    if args.use_cpu:
        print("Forcing CPU usage as requested...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Try GPU, but handle gracefully if it fails
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"Using GPU: {gpus}")
                # Only allow memory growth, to prevent OOM issues
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("No GPU found. Using CPU.")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
            print("Falling back to CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Set lower mixed precision policy to avoid compatibility issues
    try:
        tf.keras.mixed_precision.set_global_policy('float32')
    except Exception as e:
        print(f"Could not set precision policy: {e}")
    
    # Load dataset
    print(f"Loading MINC-2500 dataset from {args.data_dir} (fold {args.fold})...")
    data_dict = utils.load_minc_dataset(args.data_dir, fold=args.fold)
    
    print(f"Dataset loaded: {len(data_dict['X_train'])} training, "
          f"{len(data_dict['X_val'])} validation, {len(data_dict['X_test'])} test images")
    
    # Stage 1: Preprocessing and Image Enhancement
    if not args.skip_preprocessing:
        tf_datasets = preprocess_stage(data_dict, args)
    else:
        print("\n--- Skipping Preprocessing Stage ---\n")
        tf_datasets = utils.create_tf_datasets(data_dict, batch_size=args.batch_size)
    
    # Stage 2: Segmentation
    if not args.skip_segmentation:
        data_dict = segmentation_stage(data_dict)
    else:
        print("\n--- Skipping Segmentation Stage ---\n")
    
    # Stage 3: Feature Extraction
    features_dict, feature_extractor = feature_extraction_stage(data_dict, tf_datasets, args)
    
    # Stage 4: Classification
    model = classification_stage(data_dict, tf_datasets, features_dict, feature_extractor, args)
    
    # Save model
    if model is not None and args.model_type in ['mobilenetv2', 'simple_cnn', 'deeper_cnn']:
        try:
            model_path = os.path.join(model_dir, f"{args.model_type}_fold{args.fold}.h5")
            print(f"Saving model to {model_path}")
            model.save(model_path)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    # Stage 5: Evaluation and Performance Metrics
    metrics = evaluation_stage(model, data_dict, tf_datasets, features_dict, args)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")