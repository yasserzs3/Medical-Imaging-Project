"""Evaluation metrics for medical image segmentation."""

import torch
import numpy as np


class ConfusionMatrix:
    """
    Calculate confusion matrix for segmentation tasks
    
    This class calculates TP, TN, FP, FN and derives metrics:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Precision: TP / (TP + FP)
    - Recall/Sensitivity: TP / (TP + FN)
    - Specificity: TN / (TN + FP)
    - F1 Score: 2 * Precision * Recall / (Precision + Recall)
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """Reset counters to zero"""
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.samples = 0
        
    def update(self, pred, target):
        """
        Update confusion matrix counters
        
        Args:
            pred (torch.Tensor): Predicted tensor (raw logits)
            target (torch.Tensor): Ground truth tensor
        """
        pred = torch.sigmoid(pred) > self.threshold
        pred = pred.float()
        target = target.float()
        
        self.samples += pred.shape[0]
        
        # Flatten tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        # Calculate TP, TN, FP, FN
        self.tp += torch.sum(pred * target).item()
        self.tn += torch.sum((1 - pred) * (1 - target)).item()
        self.fp += torch.sum(pred * (1 - target)).item()
        self.fn += torch.sum((1 - pred) * target).item()
    
    def get_metrics(self):
        """
        Calculate metrics from confusion matrix
        
        Returns:
            dict: Dictionary with metrics
        """
        # Avoid division by zero
        epsilon = 1e-7
        
        # Calculate metrics
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon)
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        specificity = self.tn / (self.tn + self.fp + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'iou': iou
        }
        
    def __str__(self):
        """String representation of metrics"""
        metrics = self.get_metrics()
        return (f"Samples: {self.samples}\n"
                f"TP: {self.tp}, TN: {self.tn}, FP: {self.fp}, FN: {self.fn}\n"
                f"Accuracy: {metrics['accuracy']:.4f}\n"
                f"Precision: {metrics['precision']:.4f}\n"
                f"Recall: {metrics['recall']:.4f}\n"
                f"Specificity: {metrics['specificity']:.4f}\n"
                f"F1 Score: {metrics['f1']:.4f}\n"
                f"IoU: {metrics['iou']:.4f}")


def accuracy(y_pred, y_true):
    """
    Calculate pixel-wise accuracy for predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        
    Returns:
        float: Accuracy score
    """
    # Ensure predictions are binary (0 or 1)
    y_pred = (y_pred > 0.5).float()
    
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate correct predictions
    correct = (y_pred_flat == y_true_flat).float().sum()
    total = y_true_flat.numel()
    
    return (correct / total).item()


def sensitivity(y_pred, y_true, smooth=1.0):
    """
    Calculate sensitivity (recall, true positive rate).
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Sensitivity score
    """
    # Ensure predictions are binary (0 or 1)
    y_pred = (y_pred > 0.5).float()
    
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate true positives
    true_positives = (y_pred_flat * y_true_flat).sum()
    actual_positives = y_true_flat.sum()
    
    # Calculate sensitivity
    sens = (true_positives + smooth) / (actual_positives + smooth)
    
    return sens.item()


def specificity(y_pred, y_true, smooth=1.0):
    """
    Calculate specificity (true negative rate).
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Specificity score
    """
    # Ensure predictions are binary (0 or 1)
    y_pred = (y_pred > 0.5).float()
    
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate true negatives
    true_negatives = ((1 - y_pred_flat) * (1 - y_true_flat)).sum()
    actual_negatives = (1 - y_true_flat).sum()
    
    # Calculate specificity
    spec = (true_negatives + smooth) / (actual_negatives + smooth)
    
    return spec.item()


def calculate_metrics(y_pred, y_true):
    """
    Calculate all metrics for predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Create a confusion matrix
    cm = ConfusionMatrix(threshold=0.5)
    cm.update(y_pred, y_true)
    
    # Return all metrics
    return cm.get_metrics() 