# train_resnet.py
"""
ResNet-18 training with improved generalization for CIFAR-10
Using techniques to prevent overfitting:
1. Data augmentation
2. Dropout
3. Label smoothing
4. Stronger regularization
5. Gradual learning rate warmup
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from src.cifar_split import setup_cifar10
from tqdm import tqdm
import numpy as np

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_resnet():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Hyperparameters
    num_epochs = 200
    batch_size = 128
    initial_lr = 0.01
    weight_decay = 1e-4
    dropout_rate = 0.3
    label_smoothing = 0.1
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model directory
    model_dir = '/ssd_4TB/divake/learnable_scoring_funtion_01/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Define stronger data augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets with custom transforms
    train_loader, val_loader, test_loader, _, _, _ = setup_cifar10(
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform
    )
    
    # Create model with dropout
    model = models.resnet18(weights=None)
    # Add dropout layers
    model.layer1 = nn.Sequential(model.layer1, nn.Dropout2d(dropout_rate))
    model.layer2 = nn.Sequential(model.layer2, nn.Dropout2d(dropout_rate))
    model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(dropout_rate))
    model.layer4 = nn.Sequential(model.layer4, nn.Dropout2d(dropout_rate))
    
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, 10)
    )
    
    model.apply(init_weights)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = LabelSmoothingLoss(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, 
                           weight_decay=weight_decay, amsgrad=True)
    
    # Learning rate scheduler with warmup
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=15)
    best_val_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/train_total:.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_loss = train_loss/train_total
        train_acc = 100.*train_correct/train_total
        val_loss = val_loss/val_total
        val_acc = 100.*val_correct/val_total
        
        # Print results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            print(f'Saving best model with validation accuracy: {val_acc:.2f}%')
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'resnet18_cifar10_best.pth'))
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Final evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
    
    test_acc = 100.*test_correct/test_total
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

if __name__ == "__main__":
    train_resnet()