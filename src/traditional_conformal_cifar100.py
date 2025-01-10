import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import timm
import torchvision.transforms as transforms
from cifar_split import setup_cifar100

class RAPSConformalPredictor:
    def __init__(self, base_model, device, coverage_target=0.9, kreg=3, lamda=0.15, alpha=0.1):
        self.base_model = base_model
        self.device = device
        self.coverage_target = coverage_target
        self.num_classes = 100
        self.kreg = kreg
        self.lamda = lamda
        self.alpha = alpha
        self.tau = None
        
    def compute_penalty(self, size, scores):
        """Compute adaptive penalty with smoother growth"""
        # More gradual penalty growth
        base_size_penalty = self.lamda * (1 + np.maximum(0, size - self.kreg)**1.5 / 3)
        
        # Compute confidence-based scaling that grows more slowly
        top_3_sum = np.sum(scores[:, :3], axis=1)
        score_ratio = np.minimum(2, scores[:, 0] / (scores[:, 1] + 1e-6))  # Cap ratio at 2
        
        # Gentler confidence factor
        confidence_factor = (1 + self.alpha * np.log1p(score_ratio - 1)) * (1 + 0.5 * top_3_sum)
        
        return np.outer(confidence_factor, base_size_penalty)

    def calibrate(self, cal_loader):
        """Compute tau using calibration set with more aggressive coverage control"""
        self.base_model.eval()
        all_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(cal_loader, desc="Calibrating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                scores = probs.cpu().numpy()
                
                I = np.argsort(-scores, axis=1)
                ordered = np.take_along_axis(scores, I, axis=1)
                cumsum = np.cumsum(ordered, axis=1)
                
                # Compute adaptive penalties
                penalties = self.compute_penalty(np.arange(1, self.num_classes + 1), ordered)
                
                for i, target in enumerate(targets.cpu().numpy()):
                    target_idx = np.where(I[i] == target)[0][0]
                    if target_idx == 0:
                        score = 1 - scores[i, target]
                    else:
                        score = 1 - (cumsum[i, target_idx] + penalties[i, target_idx])
                    all_scores.append(score)
        
        # Use aggressive quantile adjustment
        target_quantile = 1 - (self.coverage_target - 0.1)  # Aim lower to account for conservativeness
        self.tau = np.quantile(all_scores, target_quantile)
        return self.tau
    
    def evaluate(self, loader):
        """Evaluate with adaptive RAPS prediction sets"""
        if self.tau is None:
            raise ValueError("Must calibrate first!")
            
        self.base_model.eval()
        coverages = []
        set_sizes = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                scores = probs.cpu().numpy()
                
                I = np.argsort(-scores, axis=1)
                ordered = np.take_along_axis(scores, I, axis=1)
                cumsum = np.cumsum(ordered, axis=1)
                
                # Compute penalties
                penalties = self.compute_penalty(np.arange(1, self.num_classes + 1), ordered)
                
                for i, target in enumerate(targets.cpu().numpy()):
                    k = 1
                    while k <= self.num_classes:
                        if cumsum[i, k-1] + penalties[i, k-1] > 1 - self.tau:
                            break
                        k += 1
                    k = max(1, k-1)
                    
                    pred_set = set(I[i, :k])
                    set_sizes.append(k)
                    coverages.append(int(target in pred_set))
        
        # Calculate statistics
        coverage = np.mean(coverages)
        avg_size = np.mean(set_sizes)
        max_size = np.max(set_sizes)
        min_size = np.min(set_sizes)
        size_std = np.std(set_sizes)
        
        unique_sizes, size_counts = np.unique(set_sizes, return_counts=True)
        size_dist = {size: count/len(set_sizes) for size, count in zip(unique_sizes, size_counts)}
        
        print(f"\nDetailed Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Set Size Statistics:")
        print(f"  - Average: {avg_size:.4f}")
        print(f"  - Std Dev: {size_std:.4f}")
        print(f"  - Min: {min_size}")
        print(f"  - Max: {max_size}")
        print("\nSet Size Distribution:")
        for size in sorted(size_dist.keys()):
            print(f"  Size {size}: {size_dist[size]*100:.1f}%")
        
        return {
            'coverage': coverage,
            'avg_set_size': avg_size,
            'max_set_size': max_size,
            'min_set_size': min_size,
            'size_std': size_std,
            'size_distribution': size_dist
        }
def get_transforms():
    """Get the transforms used during ViT training"""
    return transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = '/ssd_4TB/divake/learnable_scoring_fn'
    
    # Load data with transforms
    transform = get_transforms()
    _, cal_loader, test_loader, _, cal_dataset, test_dataset = setup_cifar100(batch_size=128)
    
    # Update transforms
    if hasattr(cal_dataset, 'dataset'):
        cal_dataset.dataset.transform = transform
        test_dataset.transform = transform
    else:
        cal_dataset.transform = transform
        test_dataset.transform = transform
    
    # Load pretrained ViT model
    base_model = timm.create_model(
        'vit_base_patch16_224_in21k',
        pretrained=False,
        num_classes=100,
        img_size=96,
        drop_path_rate=0.1,
        drop_rate=0.1
    )
    base_model.load_state_dict(torch.load(os.path.join(base_path, 'models/vit_phase2_best.pth')))
    base_model = base_model.to(device)
    base_model.eval()
    
    # Try different combinations of parameters
    params_to_try = [
        {'kreg': 3, 'lamda': 0.05, 'alpha': 0.1},  # Conservative
        {'kreg': 4, 'lamda': 0.03, 'alpha': 0.15},  # Balanced
        {'kreg': 3, 'lamda': 0.07, 'alpha': 0.2}   # Aggressive
    ]
    
    for params in params_to_try:
        print(f"\nTrying parameters:")
        print(f"kreg={params['kreg']}, lamda={params['lamda']}, alpha={params['alpha']}")
        predictor = RAPSConformalPredictor(
            base_model, 
            device,
            **params
        )
        
        tau = predictor.calibrate(cal_loader)
        print(f"Calibrated tau: {tau:.4f}")
        results = predictor.evaluate(test_loader)

if __name__ == "__main__":
    main()