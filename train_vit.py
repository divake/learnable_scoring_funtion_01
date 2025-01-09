#train_vit.py

from torchvision.datasets import CIFAR100
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
from PIL import Image
from src.cifar_split import setup_cifar100
import os

MODEL_SAVE_DIR = '/ssd_4TB/divake/learnable_scoring_fn/models'

def mixup_data(x, y, device='cuda', alpha=0.8):
    """Performs mixup on the input data and label vectors."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def train_vit():
    """Main training function with two-phase training approach"""
    # Device setup
    torch.cuda.set_device(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    
    # Create model with correct image size configuration
    model = timm.create_model(
        'vit_base_patch16_224_in21k',
        pretrained=True,
        num_classes=100,
        img_size=96,  # Reduced from 224 to 96
        drop_path_rate=0.1,
        drop_rate=0.1
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.set_grad_checkpointing(enable=True)
    
    # Freeze base layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Only train the head
    for param in model.head.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Load and setup datasets
    train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar100(
        batch_size=batch_size
    )
    
    # Update transforms
    train_dataset.dataset.transform = train_transform
    cal_dataset.dataset.transform = test_transform
    test_dataset.transform = test_transform
    
    # Recreate data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    cal_loader = DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Phase 1: Training only the top layers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.01,
        momentum=0.9
    )
    
    print("Phase 1: Training top layers only")
    best_acc = 0
    for epoch in range(10):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        
        # Validation using calibration set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in cal_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss/len(cal_loader)
        test_acc = 100.*correct/total
        
        print(f'Phase 1 - Epoch {epoch+1}/10:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Cal Loss: {test_loss:.3f} | Cal Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'vit_phase1_best.pth'))
    
    # Phase 2: Fine-tuning the whole model
    print("\nPhase 2: Fine-tuning the whole model")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Update hyperparameters for fine-tuning
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=40,
        eta_min=1e-6
    )
    
    best_acc = 0
    for epoch in range(40):
        model.train()
        running_loss = 0
        correct_mixed = 0
        correct_clean = 0
        total_mixed = 0
        total_clean = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup augmentation
            if random.random() < 0.5:
                mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, device=device)
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                _, predicted = outputs.max(1)
                total_mixed += targets.size(0)
                correct_mixed += (lam * predicted.eq(targets_a).sum().item() + 
                                (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                _, predicted = outputs.max(1)
                total_clean += targets.size(0)
                correct_clean += predicted.eq(targets).sum().item()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss/len(train_loader)
        train_acc_mixed = 100.*correct_mixed/total_mixed if total_mixed > 0 else 0
        train_acc_clean = 100.*correct_clean/total_clean if total_clean > 0 else 0
        
        # Test phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss/len(test_loader)
        test_acc = 100.*correct/total
        
        print(f'Phase 2 - Epoch {epoch+1}/40:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc (Mixed): {train_acc_mixed:.2f}% | Train Acc (Clean): {train_acc_clean:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'vit_phase2_best.pth'))
        
        scheduler.step()
    
    print(f'Training completed! Best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_vit()
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)