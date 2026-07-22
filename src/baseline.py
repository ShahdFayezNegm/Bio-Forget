import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.trainer import ModelTrainer

# =================================================================================
# FULL RETRAINING BASELINE
# =================================================================================

class FullRetrainingBaseline:
    """
    Baseline implementation that retrains the model from scratch.
    Used to compare with SISA unlearning.
    """
    
    def __init__(self):
        self.results = {}
        self.comparison_results = {}
        print("[FullRetrainingBaseline] Initialized")

    def run_full_retraining(self, train_features, train_labels, test_features, test_labels, 
                            model_class=None, model_name=None, epochs=20, verbose=True):
        """
        Run full retraining from scratch.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
            model_class: Model class (e.g., CancerDetectionCNN1D)
            model_name: Model name string (for logging)
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            tuple: (metrics_dict, training_time_seconds, trainer_object)
        """
        # Determine model name for logging
        if model_name is None:
            model_name = model_class.__name__ if model_class else "Model"
        
        if verbose:
            print(f"\n[Full Retraining] Training {model_name} from scratch...")

        start_time = time.time()

        # Create model
        model = model_class(input_dim=train_features.shape[1])
        trainer = ModelTrainer(model, model_name=f"{model_name}_FullRetrain")

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features), 
            torch.LongTensor(train_labels)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_features), 
            torch.LongTensor(test_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=min(32, len(test_dataset)), shuffle=False)

        # Train
        metrics, _ = trainer.train(train_loader, test_loader, epochs=epochs)

        elapsed_time = time.time() - start_time

        # Calculate additional metrics if not already present
        if 'precision' not in metrics:
            # Get predictions for metrics
            model.eval()
            all_preds = []
            all_probs = []
            all_targets = []
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(trainer.device)
                    outputs = model(data)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_targets.extend(targets.numpy())
            
            metrics['precision'] = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['auc'] = roc_auc_score(all_targets, all_probs)

        # Store results
        self.results[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc': metrics.get('auc', 0),
            'training_time': elapsed_time,
            'epochs': epochs
        }

        if verbose:
            print(f"   Full retraining completed in {elapsed_time:.2f}s")
            print(f"   Accuracy: {metrics['accuracy']:.2f}%")
            print(f"   AUC: {metrics.get('auc', 0):.3f}")
            print(f"   F1: {metrics.get('f1', 0):.3f}")
        
        return metrics, elapsed_time, trainer

    def run_fast_retraining(self, train_features, train_labels, test_features, test_labels,
                            model_class=None, model_name=None, epochs=10):
        """
        Run fast retraining with fewer epochs for quick comparison.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            test_features: Test features
            test_labels: Test labels
            model_class: Model class
            model_name: Model name
            epochs: Number of epochs (default 10)
        
        Returns:
            tuple: (metrics_dict, training_time_seconds, trainer_object)
        """
        print(f"\n[Fast Retraining] Training {model_name} for {epochs} epochs...")
        return self.run_full_retraining(
            train_features, train_labels, test_features, test_labels,
            model_class=model_class, model_name=model_name, epochs=epochs, verbose=False
        )

    def compare_with_sisa(self, sisa_results, sisa_time=None):
        """
        Compare baseline results with SISA results.
        
        Args:
            sisa_results (dict): Results from SISA unlearning
            sisa_time (float): SISA unlearning time
        
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for model_name, baseline_result in self.results.items():
            # Find matching SISA result (try exact match or first available)
            sisa_result = None
            if model_name in sisa_results:
                sisa_result = sisa_results[model_name]
            elif sisa_results:
                # Use first available SISA result
                sisa_result = list(sisa_results.values())[0]
            
            if sisa_result:
                sisa_time_val = sisa_time or sisa_result.get('training_time', 0)
                
                comparison[model_name] = {
                    'baseline_accuracy': baseline_result['accuracy'],
                    'sisa_accuracy': sisa_result.get('accuracy', 0),
                    'baseline_time': baseline_result['training_time'],
                    'sisa_time': sisa_time_val,
                    'speedup': baseline_result['training_time'] / (sisa_time_val + 1e-10),
                    'accuracy_diff': baseline_result['accuracy'] - sisa_result.get('accuracy', 0),
                    'baseline_auc': baseline_result.get('auc', 0),
                    'sisa_auc': sisa_result.get('auc', 0),
                    'baseline_f1': baseline_result.get('f1', 0),
                    'sisa_f1': sisa_result.get('f1', 0)
                }
        
        self.comparison_results = comparison
        return comparison

    def compare_multiple_methods(self, method_results):
        """
        Compare multiple unlearning methods.
        
        Args:
            method_results (dict): Dictionary with method names as keys and result dicts as values
        
        Returns:
            dict: Comparison results
        """
        comparison = {}
        
        for method_name, results in method_results.items():
            comparison[method_name] = {
                'accuracy': results.get('accuracy', 0),
                'time': results.get('training_time', 0),
                'auc': results.get('auc', 0),
                'f1': results.get('f1', 0)
            }
        
        return comparison

    def print_comparison(self, comparison=None, title="BASELINE VS SISA COMPARISON"):
        """
        Print comparison between baseline and SISA.
        
        Args:
            comparison (dict): Comparison results from compare_with_sisa()
            title (str): Title for the comparison table
        """
        if comparison is None:
            comparison = self.comparison_results
        
        if not comparison:
            print("No comparison data available.")
            return
        
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
        print(f"{'Model':<20} {'Baseline Acc':<15} {'SISA Acc':<15} {'Speedup':<10} {'Acc Diff':<10}")
        print("-"*80)
        
        for model_name, comp in comparison.items():
            acc_diff = comp.get('accuracy_diff', comp['baseline_accuracy'] - comp.get('sisa_accuracy', 0))
            print(f"{model_name:<20} {comp['baseline_accuracy']:.2f}%     {comp.get('sisa_accuracy', 0):.2f}%     {comp.get('speedup', 0):.2f}x    {acc_diff:+.2f}%")
        
        print("="*80)

    def print_detailed_comparison(self, comparison=None):
        """
        Print detailed comparison with AUC, F1, and time.
        
        Args:
            comparison (dict): Comparison results from compare_with_sisa()
        """
        if comparison is None:
            comparison = self.comparison_results
        
        if not comparison:
            print("No comparison data available.")
            return
        
        print("\n" + "="*90)
        print(" DETAILED COMPARISON: FULL RETRAINING vs SISA")
        print("="*90)
        print(f"{'Model':<18} {'Method':<15} {'Accuracy':<12} {'AUC':<10} {'F1':<10} {'Time (s)':<10}")
        print("-"*90)
        
        for model_name, comp in comparison.items():
            print(f"{model_name:<18} {'Full Retrain':<15} {comp['baseline_accuracy']:.2f}%    {comp.get('baseline_auc', 0):.3f}    {comp.get('baseline_f1', 0):.3f}    {comp['baseline_time']:.2f}")
            print(f"{model_name:<18} {'SISA':<15} {comp.get('sisa_accuracy', 0):.2f}%    {comp.get('sisa_auc', 0):.3f}    {comp.get('sisa_f1', 0):.3f}    {comp.get('sisa_time', 0):.2f}")
            print("-"*90)
        
        print("="*90)

    def get_summary(self):
        """
        Get summary of all results.
        
        Returns:
            dict: Summary dictionary
        """
        return {
            'results': self.results,
            'comparison': self.comparison_results
        }

    def save_results(self, filepath="baseline_results.json"):
        """
        Save results to JSON file.
        
        Args:
            filepath (str): Path to save results
        """
        import json
        import numpy as np
        
        # Convert numpy values to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        results_copy = convert_to_serializable({
            'results': self.results,
            'comparison': self.comparison_results
        })
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2)
            print(f" Results saved to {filepath}")
        except Exception as e:
            print(f" Could not save results: {e}")