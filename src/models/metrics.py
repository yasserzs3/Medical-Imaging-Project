import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_coefficient(y_pred, y_true, smooth=1.0):
    """
    Calculate Dice coefficient for predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    # Ensure predictions are binary (0 or 1)
    y_pred = (y_pred > 0.5).float()
    
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def iou_score(y_pred, y_true, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score for predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        float: IoU score
    """
    # Ensure predictions are binary (0 or 1)
    y_pred = (y_pred > 0.5).float()
    
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate intersection and union
    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


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
    Calculate sensitivity (true positive rate) for predictions.
    
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
    Calculate specificity (true negative rate) for predictions.
    
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


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)
        
        # Flatten the predictions and targets
        batch_size = probs.size(0)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Calculate Dice coefficient
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        # Calculate Dice loss
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        return dice_loss.mean()


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, logits, targets):
        bce_loss = self.bce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        combined_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return combined_loss


def calculate_metrics(y_pred, y_true):
    """
    Calculate all metrics for predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted masks (after sigmoid if using logits)
        y_true (torch.Tensor): Ground truth masks
        
    Returns:
        dict: Dictionary containing all metrics
    """
    return {
        'dice': dice_coefficient(y_pred, y_true),
        'iou': iou_score(y_pred, y_true),
        'accuracy': accuracy(y_pred, y_true),
        'sensitivity': sensitivity(y_pred, y_true),
        'specificity': specificity(y_pred, y_true)
    }


def dice_coef(pred, target, threshold=0.5, smooth=1.0):
    """
    Calculate Dice coefficient
    
    Args:
        pred (torch.Tensor): Predicted tensor (raw logits)
        target (torch.Tensor): Ground truth tensor
        threshold (float): Threshold for binary segmentation
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        float: Dice coefficient
    """
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred) > threshold
    pred = pred.float()
    
    # Check if both prediction and target are empty
    all_zeros_pred = (pred.sum() == 0)
    all_zeros_target = (target.sum() == 0)
    
    if all_zeros_pred and all_zeros_target:
        # If both are empty, dice should be 1
        return 1.0
    
    # Move to 1D tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and dice
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice loss for segmentation
    
    Args:
        pred (torch.Tensor): Predicted tensor (raw logits)
        target (torch.Tensor): Ground truth tensor
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        torch.Tensor: Dice loss
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1.0 - dice


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Calculate IoU (Intersection over Union) score
    
    Args:
        pred (torch.Tensor): Predicted tensor (raw logits)
        target (torch.Tensor): Ground truth tensor
        threshold (float): Threshold for binary segmentation
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        float: IoU score
    """
    # Apply sigmoid and threshold
    pred = torch.sigmoid(pred) > threshold
    pred = pred.float()
    
    # Check if both prediction and target are empty
    all_zeros_pred = (pred.sum() == 0)
    all_zeros_target = (target.sum() == 0)
    
    if all_zeros_pred and all_zeros_target:
        # If both are empty, IoU should be 1
        return 1.0
    
    # Move to 1D tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def iou_loss(pred, target, smooth=1.0):
    """
    Calculate IoU loss for segmentation
    
    Args:
        pred (torch.Tensor): Predicted tensor (raw logits)
        target (torch.Tensor): Ground truth tensor
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        torch.Tensor: IoU loss
    """
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return 1.0 - iou


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        return dice_loss(pred, target, self.smooth)


class IoULoss(nn.Module):
    """
    IoU Loss for segmentation
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        return iou_loss(pred, target, self.smooth)


class ComboLoss(nn.Module):
    """
    Combined loss: BCE + Dice + IoU
    
    Args:
        weights (list): Weights for BCE, Dice, and IoU losses
        smooth (float): Smoothing constant
    """
    def __init__(self, weights=[0.3, 0.4, 0.3], smooth=1.0):
        super(ComboLoss, self).__init__()
        self.weights = weights
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)
        self.iou_loss = IoULoss(smooth)
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        return self.weights[0] * bce_loss + self.weights[1] * dice + self.weights[2] * iou 


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


def multiclass_dice_coef(pred, target, num_classes=2, threshold=0.5, smooth=1.0):
    """
    Calculate Dice coefficient for multi-class segmentation
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, C, H, W)
        target (torch.Tensor): Ground truth tensor of shape (B, C, H, W) or (B, 1, H, W)
        num_classes (int): Number of classes
        threshold (float): Threshold for binary segmentation
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        list: List of Dice coefficients for each class
        float: Mean Dice coefficient
    """
    # For binary segmentation with output shape (B, 1, H, W)
    if pred.shape[1] == 1 and num_classes == 2:
        # Assuming binary case where class 0 is implicit
        dice_class0 = dice_coef(1.0 - pred, 1.0 - target, threshold, smooth)
        dice_class1 = dice_coef(pred, target, threshold, smooth)
        return [dice_class0, dice_class1], (dice_class0 + dice_class1) / 2.0
    
    # For multi-class segmentation
    dice_scores = []
    
    # Convert target to one-hot if it's a class index tensor
    if target.shape[1] == 1:
        target_one_hot = torch.zeros_like(pred)
        for cls in range(num_classes):
            target_one_hot[:, cls, ...] = (target[:, 0, ...] == cls).float()
        target = target_one_hot
    
    # Calculate dice coefficient for each class
    for cls in range(num_classes):
        pred_cls = pred[:, cls, ...]
        target_cls = target[:, cls, ...]
        
        dice_cls = dice_coef(pred_cls, target_cls, threshold, smooth)
        dice_scores.append(dice_cls)
    
    # Mean dice coefficient
    mean_dice = sum(dice_scores) / len(dice_scores)
    
    return dice_scores, mean_dice


def multiclass_iou_score(pred, target, num_classes=2, threshold=0.5, smooth=1.0):
    """
    Calculate IoU score for multi-class segmentation
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, C, H, W)
        target (torch.Tensor): Ground truth tensor of shape (B, C, H, W) or (B, 1, H, W)
        num_classes (int): Number of classes
        threshold (float): Threshold for binary segmentation
        smooth (float): Smoothing constant to avoid division by zero
        
    Returns:
        list: List of IoU scores for each class
        float: Mean IoU score (mIoU)
    """
    # For binary segmentation with output shape (B, 1, H, W)
    if pred.shape[1] == 1 and num_classes == 2:
        # Assuming binary case where class 0 is implicit
        iou_class0 = iou_score(1.0 - pred, 1.0 - target, threshold, smooth)
        iou_class1 = iou_score(pred, target, threshold, smooth)
        return [iou_class0, iou_class1], (iou_class0 + iou_class1) / 2.0
    
    # For multi-class segmentation
    iou_scores = []
    
    # Convert target to one-hot if it's a class index tensor
    if target.shape[1] == 1:
        target_one_hot = torch.zeros_like(pred)
        for cls in range(num_classes):
            target_one_hot[:, cls, ...] = (target[:, 0, ...] == cls).float()
        target = target_one_hot
    
    # Calculate IoU score for each class
    for cls in range(num_classes):
        pred_cls = pred[:, cls, ...]
        target_cls = target[:, cls, ...]
        
        iou_cls = iou_score(pred_cls, target_cls, threshold, smooth)
        iou_scores.append(iou_cls)
    
    # Mean IoU (mIoU)
    mean_iou = sum(iou_scores) / len(iou_scores)
    
    return iou_scores, mean_iou


class MulticlassConfusionMatrix:
    """
    Calculate confusion matrix for multi-class segmentation tasks
    """
    def __init__(self, num_classes=2, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """Reset counters to zero"""
        # Initialize confusion matrix of shape (num_classes, num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.samples = 0
        
    def update(self, pred, target):
        """
        Update confusion matrix counters
        
        Args:
            pred (torch.Tensor): Predicted tensor (raw logits) of shape (B, C, H, W)
            target (torch.Tensor): Ground truth tensor of shape (B, 1, H, W) or (B, C, H, W)
        """
        self.samples += pred.shape[0]
        
        # For binary segmentation
        if pred.shape[1] == 1 and self.num_classes == 2:
            # Convert to class indices (0 and 1)
            pred_cls = (torch.sigmoid(pred) > self.threshold).long().cpu().numpy()
            if target.shape[1] == 1:
                target_cls = target.long().cpu().numpy()
            else:
                # If target is one-hot, convert to class indices
                target_cls = torch.argmax(target, dim=1, keepdim=True).cpu().numpy()
            
            # Flatten predictions and targets
            pred_cls = pred_cls.flatten()
            target_cls = target_cls.flatten()
            
            # Update confusion matrix
            for t, p in zip(target_cls, pred_cls):
                self.confusion_matrix[t, p] += 1
            
        else:
            # For multi-class segmentation
            # Apply softmax and get class indices
            if pred.shape[1] > 1:
                pred_cls = torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy()
            else:
                pred_cls = (torch.sigmoid(pred) > self.threshold).long().cpu().numpy()
                
            # Get target class indices
            if target.shape[1] == 1:
                target_cls = target.squeeze(1).long().cpu().numpy()
            else:
                target_cls = torch.argmax(target, dim=1).cpu().numpy()
            
            # Flatten predictions and targets
            pred_cls = pred_cls.flatten()
            target_cls = target_cls.flatten()
            
            # Update confusion matrix
            for t, p in zip(target_cls, pred_cls):
                self.confusion_matrix[t, p] += 1
    
    def get_metrics(self):
        """
        Calculate metrics from confusion matrix
        
        Returns:
            dict: Dictionary with metrics
        """
        # Avoid division by zero
        epsilon = 1e-7
        metrics = {}
        
        # Per-class metrics
        metrics['per_class'] = {}
        for i in range(self.num_classes):
            # For each class i, treat it as positive and all others as negative
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            tn = np.sum(self.confusion_matrix) - tp - fp - fn
            
            # Calculate metrics for class i
            accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            specificity = tn / (tn + fp + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            iou = tp / (tp + fp + fn + epsilon)
            
            metrics['per_class'][i] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'iou': iou
            }
        
        # Overall metrics (macro average)
        metrics['macro_avg'] = {
            'accuracy': np.mean([metrics['per_class'][i]['accuracy'] for i in range(self.num_classes)]),
            'precision': np.mean([metrics['per_class'][i]['precision'] for i in range(self.num_classes)]),
            'recall': np.mean([metrics['per_class'][i]['recall'] for i in range(self.num_classes)]),
            'specificity': np.mean([metrics['per_class'][i]['specificity'] for i in range(self.num_classes)]),
            'f1': np.mean([metrics['per_class'][i]['f1'] for i in range(self.num_classes)]),
            'iou': np.mean([metrics['per_class'][i]['iou'] for i in range(self.num_classes)])
        }
        
        # For binary classification, add background and foreground metrics
        if self.num_classes == 2:
            metrics['background'] = metrics['per_class'][0]
            metrics['foreground'] = metrics['per_class'][1]
            
        return metrics
        
    def __str__(self):
        """String representation of metrics"""
        metrics = self.get_metrics()
        output = f"Samples: {self.samples}\n"
        output += f"Confusion Matrix:\n{self.confusion_matrix}\n\n"
        
        # Add per-class metrics
        for i in range(self.num_classes):
            output += f"Class {i} metrics:\n"
            class_metrics = metrics['per_class'][i]
            output += f"  Accuracy: {class_metrics['accuracy']:.4f}\n"
            output += f"  Precision: {class_metrics['precision']:.4f}\n"
            output += f"  Recall: {class_metrics['recall']:.4f}\n"
            output += f"  Specificity: {class_metrics['specificity']:.4f}\n"
            output += f"  F1 Score: {class_metrics['f1']:.4f}\n"
            output += f"  IoU: {class_metrics['iou']:.4f}\n\n"
        
        # Add macro average metrics
        output += f"Macro Average metrics:\n"
        avg_metrics = metrics['macro_avg']
        output += f"  Accuracy: {avg_metrics['accuracy']:.4f}\n"
        output += f"  Precision: {avg_metrics['precision']:.4f}\n"
        output += f"  Recall: {avg_metrics['recall']:.4f}\n"
        output += f"  Specificity: {avg_metrics['specificity']:.4f}\n"
        output += f"  F1 Score: {avg_metrics['f1']:.4f}\n"
        output += f"  IoU (mIoU): {avg_metrics['iou']:.4f}\n"
        
        return output 