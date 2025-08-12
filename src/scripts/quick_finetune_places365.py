import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
from tqdm import tqdm
import json
import time
from datetime import datetime

class Places365Dataset(Dataset):
    def __init__(self, root_dir, list_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path, label = parts
                    if img_path.startswith('/'):
                        img_path = img_path[1:]
                    self.samples.append((img_path, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class Places365ValDataset(Dataset):
    def __init__(self, img_dir, list_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, label = parts
                    self.samples.append((img_name, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        full_path = os.path.join(self.img_dir, img_name)
        img = Image.open(full_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Top-1 accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = logits.topk(5, dim=1)
            for i in range(labels.size(0)):
                if labels[i] in top5_pred[i]:
                    top5_correct += 1
    
    accuracy = 100 * correct / total
    top5_accuracy = 100 * top5_correct / total
    return accuracy, top5_accuracy

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        if i % 10 == 0:
            acc = 100 * correct / total
            pbar.set_postfix({'loss': running_loss/(i+1), 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    # Configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    num_epochs = 5
    learning_rate = 5e-5
    num_workers = 4
    subset_fraction = 0.1  # Use 10% of training data for quick fine-tuning
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Training subset: {subset_fraction*100}%")
    
    # Paths
    data_dir = '/ssd_4TB/divake/learnable_scoring_funtion_01/data/places365_small'
    train_img_dir = os.path.join(data_dir, 'data_256_standard')
    val_img_dir = os.path.join(data_dir, 'val_256')
    train_list_file = os.path.join(data_dir, 'places365_train_standard.txt')
    val_list_file = os.path.join(data_dir, 'places365_val.txt')
    
    # Model save path
    model_save_dir = '/ssd_4TB/divake/learnable_scoring_funtion_01/models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    full_train_dataset = Places365Dataset(
        root_dir=train_img_dir,
        list_file=train_list_file,
        transform=train_transform
    )
    
    # Use subset for faster training
    subset_size = int(len(full_train_dataset) * subset_fraction)
    train_dataset = torch.utils.data.Subset(full_train_dataset, range(subset_size))
    
    val_dataset = Places365ValDataset(
        img_dir=val_img_dir,
        list_file=val_list_file,
        transform=val_transform
    )
    
    print(f"Training samples: {len(train_dataset)} (of {len(full_train_dataset)} total)")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Load model - use ImageNet pretrained ViT
    print("\nLoading ViT model...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=365,
        ignore_mismatched_sizes=True
    )
    
    # Initialize the new classifier head properly
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) // 2  # Half epoch warmup
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Initial evaluation
    print("\nInitial evaluation...")
    initial_acc, initial_top5 = evaluate(model, val_loader, device, "Initial eval")
    print(f"Initial - Top-1: {initial_acc:.2f}%, Top-5: {initial_top5:.2f}%")
    
    # Training loop
    best_acc = initial_acc
    training_history = []
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        
        # Evaluate
        val_acc, val_top5 = evaluate(model, val_loader, device, f"Validation epoch {epoch}")
        
        # Log results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Top-1: {val_acc:.2f}%, Val Top-5: {val_top5:.2f}%")
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_top5': val_top5
        })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = os.path.join(model_save_dir, 'vit_places365_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'top5_accuracy': val_top5,
            }, model_path)
            print(f"✓ Saved best model (acc: {val_acc:.2f}%)")
        
        # Early stopping if we reach target
        if val_acc >= 75:
            print(f"\n✓ Reached target accuracy of 75%!")
            break
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, 'vit_places365_final.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': val_acc,
        'top5_accuracy': val_top5,
        'training_history': training_history
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(model_save_dir, 'places365_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Models saved to: {model_save_dir}")
    
    return best_acc

if __name__ == "__main__":
    accuracy = main()
    print(f"\nFinal best accuracy: {accuracy:.2f}%")