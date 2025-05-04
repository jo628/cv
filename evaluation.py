"""
Evaluation Module

This module contains functions for evaluating classification models:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC curves
- Cross-validation
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def calculate_metrics(y_true, y_pred, y_prob=None, average='weighted'):
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Probability predictions (optional)
        average: Type of averaging for metrics
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Print a classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: List of class names
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, figsize=(10, 8)):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the matrix
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_roc_curve(y_true, y_prob, n_classes):
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_prob: Probability predictions
        n_classes: Number of classes
    """
    # Convert to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot each class's ROC curve
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, n_classes):
    """
    Plot precision-recall curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_prob: Probability predictions
        n_classes: Number of classes
    """
    # Convert to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_onehot.ravel(), y_prob.ravel())
    average_precision["micro"] = average_precision_score(y_true_onehot.ravel(), y_prob.ravel())
    
    # Plot precision-recall curves
    plt.figure(figsize=(10, 8))
    
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average precision-recall (AP = {average_precision["micro"]:.2f})',
             color='gold', linestyle=':', linewidth=4)
    
    # Plot each class's PR curve
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label=f'Precision-recall for class {i} (AP = {average_precision[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.show()


def perform_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Perform cross-validation and return scores.
    
    Args:
        model: Model to evaluate
        X: Feature data
        y: Target labels
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Cross-validation scores
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"{cv}-fold Cross Validation {scoring.capitalize()} Scores: {cv_scores}")
    print(f"Mean {scoring.capitalize()}: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    
    return cv_scores


def plot_learning_curves(history):
    """
    Plot learning curves from training history.
    
    Args:
        history: Training history object
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_model_comparison(model_names, accuracies, metric_name='Accuracy'):
    """
    Plot comparison of different models.
    
    Args:
        model_names: List of model names
        accuracies: List of accuracy values
        metric_name: Name of the metric being compared
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Model Comparison by {metric_name}')
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.ylim(0, 1.1)  # For percentage metrics
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(model, X_test, y_test, class_names=None):
    """
    Comprehensive evaluation of a model's performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Names of the classes
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)
    else:
        y_prob = model.predict(X_test)
        y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print classification report
    print("Classification Report:")
    print_classification_report(y_test, y_pred, target_names=class_names)
    
    # Plot confusion matrix
    if class_names is not None:
        print("Confusion Matrix:")
        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)
    
    # Plot ROC curve if we have probability predictions
    if y_prob is not None and y_prob.shape[1] > 1:
        n_classes = y_prob.shape[1]
        print("ROC Curves:")
        plot_roc_curve(y_test, y_prob, n_classes)
        
        print("Precision-Recall Curves:")
        plot_precision_recall_curve(y_test, y_prob, n_classes)
    
    return metrics