"""Loss functions for medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_coef(pred, target, threshold=0.5, smooth=1e-6):
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
        return torch.tensor(1.0, device=pred.device)
    
    # Move to 1D tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and dice
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def dice_loss(pred, target, smooth=1e-6):
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


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
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
        return torch.tensor(1.0, device=pred.device)
    
    # Move to 1D tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def iou_loss(pred, target, smooth=1e-6):
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
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        return dice_loss(pred, target, self.smooth)


class IoULoss(nn.Module):
    """
    IoU Loss for segmentation
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        return iou_loss(pred, target, self.smooth)


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class ComboLoss(nn.Module):
    """
    Combined loss: BCE + Dice + IoU
    
    Args:
        weights (list): Weights for BCE, Dice, and IoU losses
        smooth (float): Smoothing constant
    """
    def __init__(self, weights=[0.3, 0.4, 0.3], smooth=1e-6):
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