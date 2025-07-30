#!/usr/bin/env python3
"""
Quick script to check ResNet18 accuracy on CIFAR-100 dataset
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import numpy as np

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def main():
    # Configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms (matching the config)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # ResNet uses original CIFAR size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Load CIFAR-100 test dataset
    print("Loading CIFAR-100 test dataset...")
    test_dataset = torchvision.datasets.CIFAR100(
        root='data/cifar100',
        train=False,
        download=False,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load ResNet18 model with CIFAR-specific architecture
    print("Loading ResNet18 model...")
    # Use the CIFAR-specific ResNet18 from torchvision
    from torchvision import models
    model = models.resnet18(pretrained=False, num_classes=100)
    
    # Modify the first conv layer to match CIFAR architecture (3x3 instead of 7x7)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()  # Remove max pooling for CIFAR
    
    # Load pretrained weights
    checkpoint_path = 'models/resnet18_cifar100_hf.pth'
    print(f"Loading weights from {checkpoint_path}...")
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print("Successfully loaded pretrained weights")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    print("\nEvaluating model on CIFAR-100 test set...")
    accuracy, correct, total = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*50}")
    print(f"ResNet18 CIFAR-100 Test Results:")
    print(f"{'='*50}")
    print(f"Total test samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")
    
    # Also check top-5 accuracy
    print("\nCalculating top-5 accuracy...")
    model.eval()
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Top-5 Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get top 5 predictions
            _, predicted = outputs.topk(5, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
            
            top5_correct += correct[:5].reshape(-1).float().sum(0).item()
            total += labels.size(0)
    
    top5_accuracy = 100 * top5_correct / total
    print(f"\nTop-5 Accuracy: {top5_accuracy:.2f}%")

if __name__ == "__main__":
    main()