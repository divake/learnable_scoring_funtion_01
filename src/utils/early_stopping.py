# src/utils/early_stopping.py

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_metrics = None
        self.early_stop = False
        self.prev_set_sizes = []  # Track set size history
        
    def __call__(self, loss, coverage, set_size):
        # Don't stop in first few epochs
        if len(self.prev_set_sizes) < 5:
            self.prev_set_sizes.append(set_size)
            return
            
        self.prev_set_sizes.append(set_size)
        if len(self.prev_set_sizes) > 5:
            self.prev_set_sizes.pop(0)
        
        # Stop if set size is too large after initial epochs
        if set_size > 5.0 and len(self.prev_set_sizes) >= 5:
            self.early_stop = True
            return
            
        # Compute composite metric
        current_loss = loss + max(0, 0.9 - coverage) + 0.5 * max(0, set_size - 1.5)
        
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True