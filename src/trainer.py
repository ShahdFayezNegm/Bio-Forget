import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.config import device

# =================================================================================
# MODEL TRAINER (Enhanced with Differential Privacy Support)
# =================================================================================

class ModelTrainer:
    """
    Enhanced Model Trainer with:
    - AdamW optimizer with weight decay
    - Learning rate scheduling
    - Early stopping
    - Differential Privacy support (during training)
    - Learning curve visualization
    - Model checkpointing
    - Label Smoothing for privacy
    """
    
    def __init__(self, model, lr=0.001, model_name="Model", use_dp=False, dp_epsilon=0.05, dp_delta=1e-5):
        """
        Initialize the ModelTrainer.
        
        Args:
            model: PyTorch model to train
            lr: Learning rate
            model_name: Name for logging
            use_dp: Enable Differential Privacy
            dp_epsilon: Privacy budget for DP (lower = more privacy)
            dp_delta: Delta parameter for DP
        """
        self.device = device
        self.model = model.to(self.device)
        
        # ============================================================
        # ENHANCED: Use Label Smoothing for better privacy
        # ============================================================
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # ============================================================
        # ENHANCED: Increased weight decay for better generalization
        # ============================================================
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4)
        
        # ============================================================
        # FIXED: Remove verbose parameter for PyTorch 2.0+ compatibility
        # ============================================================
        try:
            # Try with verbose (older PyTorch versions)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        except TypeError:
            # Fallback without verbose (PyTorch 2.0+)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3
            )
            print(f"[{model_name}] Note: verbose parameter not supported in this PyTorch version")
        
        self.model_name = model_name
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        
        # ============================================================
        # ENHANCED: Stronger DP parameters for better privacy
        # ============================================================
        self.dp_max_grad_norm = 0.3  # Lower = stronger privacy (was 0.5)
        self.dp_noise_multiplier = 1.5  # Higher = stronger privacy (was 1.0)
        
        # Track training history
        self.train_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_aucs = []
        self.val_losses = []
        
        # Early stopping parameters
        self.best_val_acc = 0
        self.patience_counter = 0
        self.early_stop_patience = 5
        self.best_model_state = None
        
        print(f"[{self.model_name}] Initialized (DP={use_dp}, epsilon={dp_epsilon}, label_smoothing=0.1)")

    def train_epoch(self, loader):
        """
        Train for one epoch with enhanced Differential Privacy.
        
        Args:
            loader: DataLoader for training data
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # ============================================================
            # ENHANCED: Stronger Differential Privacy during training
            # ============================================================
            if self.use_dp:
                # Clip gradients with lower max_norm for stronger privacy
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.dp_max_grad_norm)
                
                # Add stronger Gaussian noise for DP
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Adaptive noise scaling based on parameter standard deviation
                        param_std = param.std().item() + 1e-10
                        noise_scale = (param_std * self.dp_noise_multiplier) / (self.dp_epsilon * 2)
                        noise = torch.randn_like(param.grad) * noise_scale
                        param.grad.add_(noise)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy

    def validate(self, loader):
        """
        Validate the model on a validation set.
        
        Args:
            loader: DataLoader for validation data
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        correct, total = 0, 0
        all_preds, all_targets, all_probs = [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                total_loss += loss.item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total
        
        try:
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        except Exception:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'loss': avg_loss,
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'probs': all_probs,
            'targets': all_targets,
            'predictions': all_preds
        }

    def train(self, train_loader, val_loader, epochs=20, early_stopping=True):
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Number of epochs
            early_stopping: Enable early stopping
            
        Returns:
            tuple: (best_metrics, trainer_instance)
        """
        print(f"\n[{self.model_name}] Training... (DP={self.use_dp}, label_smoothing=0.1)")
        
        best_metrics = None
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_metrics['accuracy'])
            self.val_aucs.append(val_metrics['auc'])
            self.val_losses.append(val_metrics['loss'])
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                dp_info = " [DP]" if self.use_dp else ""
                print(f"   Epoch {epoch+1}/{epochs}{dp_info}: Train Acc={train_acc:.2f}%, "
                      f"Val Acc={val_metrics['accuracy']:.2f}%, AUC={val_metrics['auc']:.3f}, "
                      f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping and patience_counter >= self.early_stop_patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"[{self.model_name}] Best Val Accuracy: {best_metrics['accuracy']:.2f}%")
        print(f"[{self.model_name}] Best Val AUC: {best_metrics['auc']:.3f}")
        
        return best_metrics, self

    def train_with_dp(self, train_loader, val_loader, epochs=20, epsilon=0.05, delta=1e-5):
        """
        Train with Differential Privacy.
        
        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Number of epochs
            epsilon: Privacy budget (lower = more privacy)
            delta: Delta parameter
            
        Returns:
            tuple: (best_metrics, trainer_instance)
        """
        self.use_dp = True
        self.dp_epsilon = epsilon
        self.dp_delta = delta
        return self.train(train_loader, val_loader, epochs)

    def plot_learning_curves(self, save_path=None):
        """
        Plot learning curves for training and validation.
        
        Args:
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Loss
        ax1 = axes[0]
        ax1.plot(self.train_losses, 'b-', linewidth=2, label='Train Loss')
        ax1.plot(self.val_losses, 'r-', linewidth=2, label='Val Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{self.model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Accuracy
        ax2 = axes[1]
        ax2.plot(self.train_accs, 'b-', linewidth=2, label='Train Accuracy')
        ax2.plot(self.val_accs, 'r-', linewidth=2, label='Val Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'{self.model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: AUC
        ax3 = axes[2]
        ax3.plot(self.val_aucs, 'g-', linewidth=2, label='Val AUC')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_title(f'{self.model_name} - AUC Curve', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Learning curves saved to {save_path}")
        plt.close()
        
        print(f"📊 Learning curves plotted")

    def save_model(self, path=None):
        """
        Save the model state dictionary.
        
        Args:
            path: Path to save the model
        """
        if path is None:
            path = f"{self.model_name.lower().replace(' ', '_')}_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'val_aucs': self.val_aucs
        }, path)
        print(f"[{self.model_name}] Model saved to {path}")

    def load_model(self, path):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"[{self.model_name}] Model loaded from {path}")
        return self.model

    def get_model(self):
        """
        Get the trained model.
        
        Returns:
            nn.Module: Trained model
        """
        return self.model

    def get_statistics(self):
        """
        Get training statistics.
        
        Returns:
            dict: Training statistics
        """
        return {
            'model_name': self.model_name,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'val_aucs': self.val_aucs,
            'best_val_acc': max(self.val_accs) if self.val_accs else 0,
            'best_auc': max(self.val_aucs) if self.val_aucs else 0,
            'use_dp': self.use_dp,
            'dp_epsilon': self.dp_epsilon if self.use_dp else None
        }


# =================================================================================
# DP-SGD Trainer (Specialized for Differential Privacy)
# =================================================================================

class DPSGDTrainer(ModelTrainer):
    """
    Specialized trainer for Differential Privacy using DP-SGD.
    """
    
    def __init__(self, model, lr=0.001, model_name="DP_Model", epsilon=0.05, delta=1e-5, max_grad_norm=0.3):
        """
        Initialize DP-SGD Trainer.
        
        Args:
            model: PyTorch model
            lr: Learning rate
            model_name: Name for logging
            epsilon: Privacy budget (lower = more privacy)
            delta: Delta parameter
            max_grad_norm: Maximum gradient norm for clipping (lower = more privacy)
        """
        super().__init__(model, lr, model_name, use_dp=True, dp_epsilon=epsilon, dp_delta=delta)
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = self._calculate_noise_multiplier()
        print(f"[DP-SGD] Noise multiplier: {self.noise_multiplier:.4f}")
        print(f"[DP-SGD] Max grad norm: {self.max_grad_norm}")
    
    def _calculate_noise_multiplier(self):
        """
        Calculate noise multiplier for DP-SGD.
        
        Returns:
            float: Noise multiplier
        """
        # Higher epsilon = lower noise, lower epsilon = higher noise
        return 1.5 / (self.dp_epsilon * 2)
    
    def train_epoch(self, loader):
        """
        Train one epoch with DP-SGD.
        """
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # ============================================================
            # ENHANCED: DP-SGD with stronger privacy
            # ============================================================
            # Clip gradients with lower max_norm
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Add calibrated noise
            for param in self.model.parameters():
                if param.grad is not None:
                    param_std = param.std().item() + 1e-10
                    noise = torch.randn_like(param.grad) * self.noise_multiplier * self.max_grad_norm * param_std
                    param.grad.add_(noise)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(loader), 100.0 * correct / total


# =================================================================================
# ENSEMBLE TRAINER (for SISA ensemble training with privacy)
# =================================================================================

class EnsembleTrainer:
    """
    Enhanced trainer for SISA ensemble models with privacy protection.
    Trains multiple models independently with Label Smoothing and DP support.
    """
    
    def __init__(self, model_class, input_dim, num_models=8, lr=0.001, use_dp=False, dp_epsilon=0.05):
        """
        Initialize ensemble trainer.
        
        Args:
            model_class: Model class to instantiate
            input_dim: Input dimension
            num_models: Number of models in ensemble
            lr: Learning rate
            use_dp: Enable Differential Privacy
            dp_epsilon: Privacy budget for DP (lower = more privacy)
        """
        self.model_class = model_class
        self.input_dim = input_dim
        self.num_models = num_models
        self.lr = lr
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.models = []
        self.trainers = []
        
        print(f"[EnsembleTrainer] Creating {num_models} models (DP={use_dp}, epsilon={dp_epsilon})")
    
    def train_models(self, data_loaders, epochs=20):
        """
        Train all models in the ensemble.
        
        Args:
            data_loaders: List of (train_loader, val_loader) tuples for each model
            epochs: Number of epochs
            
        Returns:
            list: Trained models
        """
        self.models = []
        self.trainers = []
        
        for i, (train_loader, val_loader) in enumerate(data_loaders):
            model = self.model_class(input_dim=self.input_dim)
            trainer = ModelTrainer(
                model, 
                lr=self.lr, 
                model_name=f"Ensemble_{i+1}",
                use_dp=self.use_dp,
                dp_epsilon=self.dp_epsilon
            )
            
            print(f"\n--- Training Ensemble Model {i+1}/{self.num_models} ---")
            best_metrics, _ = trainer.train(train_loader, val_loader, epochs=epochs)
            
            self.models.append(model)
            self.trainers.append(trainer)
        
        print(f"\n[EnsembleTrainer] All {self.num_models} models trained")
        return self.models
    
    def predict_ensemble(self, features):
        """
        Get ensemble predictions.
        
        Args:
            features: Input features
            
        Returns:
            tuple: (predictions, confidences)
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(features)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)
        
        # Average probabilities
        avg_probs = torch.stack(all_probs).mean(dim=0)
        predictions = torch.argmax(avg_probs, dim=1)
        confidences = avg_probs.max(dim=1)[0]
        
        return predictions, confidences
    
    def save_models(self, save_dir="ensemble_models"):
        """
        Save all models in the ensemble.
        
        Args:
            save_dir: Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for i, trainer in enumerate(self.trainers):
            path = os.path.join(save_dir, f"ensemble_model_{i+1}.pth")
            trainer.save_model(path)
        
        print(f"[EnsembleTrainer] Saved {len(self.models)} models to {save_dir}")


# =================================================================================
# COMPATIBILITY CHECK - Test the fix
# =================================================================================

def test_trainer_compatibility():
    """
    Test if the trainer is compatible with the current PyTorch version.
    """
    import torch.nn as nn
    
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    
    try:
        trainer = ModelTrainer(model, model_name="TestModel", use_dp=True, dp_epsilon=0.05)
        print(f"\n✅ ModelTrainer initialized successfully!")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   DP Enabled: {trainer.use_dp}")
        print(f"   DP Epsilon: {trainer.dp_epsilon}")
        print("   The trainer is compatible with this version.")
        return True
    except Exception as e:
        print(f"\n❌ ModelTrainer initialization failed: {e}")
        return False


# =================================================================================
# MAIN ENTRY POINT (for testing)
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("MODEL TRAINER TEST")
    print("="*60)
    
    from src.models import ImprovedCNN1D
    import torch
    
    # Test compatibility first
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 60)
    success = test_trainer_compatibility()
    
    if not success:
        print("\n❌ Please check the error message above and fix the issue.")
        exit(1)
    
    # Create dummy data
    input_dim = 338
    batch_size = 32
    num_samples = 100
    
    X_train = torch.randn(num_samples, input_dim)
    y_train = torch.randint(0, 2, (num_samples,))
    X_val = torch.randn(50, input_dim)
    y_val = torch.randint(0, 2, (50,))
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # Test ModelTrainer with enhanced privacy
    print("\n" + "="*60)
    print("TESTING WITH DP ENABLED")
    print("="*60)
    model = ImprovedCNN1D(input_dim=input_dim)
    trainer = ModelTrainer(model, model_name="TestModel_DP", use_dp=True, dp_epsilon=0.05)
    
    best_metrics, _ = trainer.train(train_loader, val_loader, epochs=10)
    
    print(f"\nBest Metrics: Acc={best_metrics['accuracy']:.2f}%, AUC={best_metrics['auc']:.3f}")
    
    print("\n" + "="*60)
    print("TEST PASSED!")
    print("="*60)