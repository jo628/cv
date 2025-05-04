"""
Classification Module

This module contains functions for classification:
- Simple classifier implementations (SVM, Random Forest)
- CNN implementation from scratch
- Fine-tuning MobileNetV2 for classification
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_svm_classifier(X_train, y_train, class_weight=None, C=1.0, kernel='rbf'):
    """
    Train an SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Class weights
        C: Regularization parameter
        kernel: Kernel type
        
    Returns:
        Trained SVM model
    """
    model = SVC(
        C=C,
        kernel=kernel,
        probability=True,
        class_weight=class_weight,
        decision_function_shape='ovr'
    )
    
    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, class_weight=None):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        class_weight: Class weights
        
    Returns:
        Trained Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    return model


def optimize_sklearn_model(X_train, y_train, model_type='svm', param_grid=None, cv=3):
    """
    Optimize a scikit-learn model using grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type: Type of model ('svm' or 'random_forest')
        param_grid: Parameter grid for grid search
        cv: Number of cross-validation folds
        
    Returns:
        Optimized model
    """
    if model_type == 'svm':
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
        base_model = SVC(probability=True)
    
    elif model_type == 'random_forest':
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        base_model = RandomForestClassifier(random_state=42)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use grid search to find the best hyperparameters
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def create_simple_cnn(input_shape=(224, 224, 3), num_classes=7):
    """
    Create a simple CNN architecture from scratch.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        
    Returns:
        CNN model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_deeper_cnn(input_shape=(224, 224, 3), num_classes=7):
    """
    Create a deeper CNN architecture from scratch.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        
    Returns:
        CNN model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth convolutional block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_mobilenetv2_classifier(input_shape=(224, 224, 3), num_classes=7, fine_tune_layers=100):
    """
    Create a classifier using MobileNetV2 as the base model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        fine_tune_layers: Number of layers to fine-tune (from the end)
        
    Returns:
        MobileNetV2-based model
    """
    # Load MobileNetV2 with pre-trained weights, excluding the top layer
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze all the layers except the last few
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:-min(fine_tune_layers, total_layers)]:
        layer.trainable = False
    
    # Create a new model with our custom top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_cnn_model(model, train_dataset, validation_dataset, epochs=20, callbacks=None):
    """
    Train a CNN model.
    
    Args:
        model: CNN model to train
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        epochs: Number of epochs
        callbacks: List of callbacks
        
    Returns:
        Training history
    """
    if callbacks is None:
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=5,
                min_lr=1e-6
            )
        ]
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history


def create_dataset(X, y, batch_size=32, is_training=True):
    """
    Create a TensorFlow dataset from numpy arrays.
    
    Args:
        X: Input data
        y: Labels
        batch_size: Batch size
        is_training: Whether this is a training dataset
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=len(X))
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset


def ensemble_predictions(models, X, weights=None):
    """
    Combine predictions from multiple models.
    
    Args:
        models: List of trained models
        X: Input data
        weights: Optional weights for each model
        
    Returns:
        Ensemble predictions
    """
    if weights is None:
        weights = [1.0] * len(models)
    
    weighted_preds = []
    
    for model, weight in zip(models, weights):
        if hasattr(model, 'predict_proba'):
            # For scikit-learn models
            preds = model.predict_proba(X)
        else:
            # For Keras models
            preds = model.predict(X)
        
        weighted_preds.append(weight * preds)
    
    # Average predictions
    ensemble_preds = np.sum(weighted_preds, axis=0) / np.sum(weights)
    
    return ensemble_preds