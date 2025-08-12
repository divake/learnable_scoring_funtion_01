import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
from tqdm import tqdm
import json
import argparse

class Places365ValDataset(Dataset):
    def __init__(self, img_dir, list_file, transform=None, processor=None):
        self.img_dir = img_dir
        self.transform = transform
        self.processor = processor
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
        
        if self.processor:
            # Use HuggingFace processor
            img = self.processor(img, return_tensors="pt")['pixel_values'].squeeze(0)
        elif self.transform:
            img = self.transform(img)
        
        return img, label

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
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
            top5_correct += sum([1 for i in range(labels.size(0)) 
                                if labels[i] in top5_pred[i]])
    
    accuracy = 100 * correct / total
    top5_accuracy = 100 * top5_correct / total
    return accuracy, top5_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, 
                       default='corenet-community/places365-224x224-vit-base',
                       help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # Paths
    data_dir = '/ssd_4TB/divake/learnable_scoring_funtion_01/data/places365_small'
    val_img_dir = os.path.join(data_dir, 'val_256')
    val_list_file = os.path.join(data_dir, 'places365_val.txt')
    
    print(f"Loading model: {args.model_name}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    try:
        # Try to load HuggingFace model
        processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModelForImageClassification.from_pretrained(args.model_name)
        model = model.to(device)
        
        # Create dataset with processor
        val_dataset = Places365ValDataset(
            img_dir=val_img_dir,
            list_file=val_list_file,
            processor=processor
        )
        
        print(f"Model loaded successfully from HuggingFace")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
    except Exception as e:
        print(f"Error loading HuggingFace model: {e}")
        print("Trying alternative approach with standard ViT...")
        
        # Use standard torchvision transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = Places365ValDataset(
            img_dir=val_img_dir,
            list_file=val_list_file,
            transform=transform
        )
        
        # Load a generic ViT model
        from transformers import ViTForImageClassification
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=365,
            ignore_mismatched_sizes=True
        )
        model = model.to(device)
        print(f"Loaded generic ViT with 365 output classes")
    
    # Create dataloader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"\nValidation set: {len(val_dataset)} images")
    print(f"Device: {device}")
    print("\nEvaluating model...")
    
    # Evaluate
    top1_acc, top5_acc = evaluate_model(model, val_loader, device)
    
    print(f"\n=== Results ===")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    # Save results
    results = {
        'model': args.model_name,
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'num_samples': len(val_dataset)
    }
    
    results_file = f"places365_eval_{args.model_name.split('/')[-1]}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return top1_acc

if __name__ == "__main__":
    accuracy = main()
    # Return status based on accuracy
    import sys
    if accuracy < 70:
        print(f"\nAccuracy {accuracy:.2f}% is below 70%, fine-tuning recommended")
        sys.exit(1)
    else:
        print(f"\nAccuracy {accuracy:.2f}% is sufficient for conformal prediction")
        sys.exit(0)