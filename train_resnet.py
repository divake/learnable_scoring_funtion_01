# train_resnet.py
"""
Improved ResNet-18 training for better accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from src.cifar_split import setup_cifar10
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):  # Increased patience and added min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_loss, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
        elif val_loss > self.best_loss - self.min_delta and val_acc <= self.best_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_acc = max(val_acc, self.best_acc)
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

def train_resnet():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    # Hyperparameters
    num_epochs = 200  # Increased max epochs
    batch_size = 128
    initial_lr = 0.1  # Increased learning rate
    patience = 25  # Increased patience
    
    # Setup device
    device = torch.device("cuda")
    
    # Create model directory
    model_dir = '/ssd1/divake/learnable_scoring_fn/models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load datasets
    train_loader, _, test_loader, _, _, _ = setup_cifar10(batch_size=batch_size)
    
    # Create model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.apply(init_weights)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, 
                         momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                             milestones=[60, 120, 160], 
                                             gamma=0.2)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    best_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, targets in pbar:
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
            
            pbar.set_postfix({
                'Loss': f'{running_loss/total:.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_loss = running_loss/total
        train_acc = 100.*correct/total
        
        # Testing phase
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
        
        test_loss = test_loss/total
        test_acc = 100.*correct/total
        
        # Print results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            print(f'Saving best model with accuracy: {test_acc:.2f}%')
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_dir, 'resnet18_cifar10_best.pth'))
        
        # Early stopping check
        early_stopping(test_loss, test_acc)
        if early_stopping.early_stop and test_acc > 85:  # Only stop if accuracy is good enough
            print("Early stopping triggered!")
            break
            
        scheduler.step()
    
    print(f'Training completed! Best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_resnet()