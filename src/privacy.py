"""
Privacy Auditor Module
Handles Membership Inference Attacks and Privacy Leakage Detection
"""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter issues

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src.config import device


class PrivacyAuditor:
    """
    Enhanced Privacy Auditor for detecting membership inference attacks.
    Uses multiple attack strategies with advanced features for comprehensive privacy analysis.
    Now includes correct Attack Advantage calculation using ROC curve.
    """
    
    def __init__(self, model):
        """
        Initialize privacy auditor.
        
        Args:
            model: The target model to audit
        """
        self.model = model
        self.device = device
        self.attack_type = 'advanced'  # 'confidence', 'entropy', 'combined', 'advanced'

    def run_attack(self, train_loader, test_loader, attack_type='advanced'):
        """
        Run comprehensive membership inference attack with improved detection.
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            attack_type: Type of attack ('confidence', 'entropy', 'combined', 'advanced')
            
        Returns:
            dict: Attack results with detailed metrics
        """
        self.model.eval()
        self.attack_type = attack_type

        # Collect comprehensive attack features
        train_feats = self._get_attack_features(train_loader)
        test_feats = self._get_attack_features(test_loader)

        if len(train_feats['confidences']) == 0 or len(test_feats['confidences']) == 0:
            return {
                "attack_advantage": 0.0,
                "attack_accuracy": 0.5,
                "attack_auc": 0.5,
                "threshold": 0.5,
                "optimal_threshold": 0.5,
                "train_conf_mean": 0,
                "test_conf_mean": 0,
                "train_entropy_mean": 0,
                "test_entropy_mean": 0,
                "privacy_risk": "UNKNOWN",
                "risk_score": 50,
                "confidence_gap": 0,
                "success": False
            }

        # ============================================================
        # FIXED: Initialize all variables before try/except
        # ============================================================
        attack_accuracy = 0.5
        attack_advantage = 0.0
        threshold = 0.5
        attack_auc = 0.5
        correct_advantage = 0.0
        correct_auc = 0.5
        success = True

        try:
            # Calculate attack metrics based on type
            if attack_type == 'confidence':
                attack_accuracy, attack_advantage, threshold, attack_auc = self._confidence_attack(
                    train_feats['confidences'], test_feats['confidences']
                )
            elif attack_type == 'entropy':
                attack_accuracy, attack_advantage, threshold, attack_auc = self._entropy_attack(
                    train_feats['entropies'], test_feats['entropies']
                )
            elif attack_type == 'combined':
                attack_accuracy, attack_advantage, threshold, attack_auc = self._combined_attack(
                    train_feats, test_feats
                )
            else:  # advanced
                attack_accuracy, attack_advantage, threshold, attack_auc = self._advanced_attack(
                    train_feats, test_feats
                )

            # ============================================================
            # FIXED: Calculate Attack Advantage correctly using ROC curve
            # ============================================================
            all_scores = np.concatenate([train_feats['confidences'], test_feats['confidences']])
            all_labels = np.concatenate([np.ones(len(train_feats['confidences'])), 
                                         np.zeros(len(test_feats['confidences']))])
            
            try:
                fpr, tpr, _ = roc_curve(all_labels, all_scores)
                correct_advantage = (tpr - fpr).max()
                correct_auc = roc_auc_score(all_labels, all_scores)
                
                print(f"\n   Corrected Attack Metrics:")
                print(f"   AUC: {correct_auc:.4f}")
                print(f"   Max TPR - FPR: {correct_advantage:.4f}")
                
                # Use the corrected values if they make sense
                if correct_advantage >= 0:
                    attack_advantage = correct_advantage
                    attack_auc = correct_auc
                
            except Exception as e:
                print(f"   ⚠️  Could not calculate ROC-based advantage: {e}")
                # Keep the existing attack_advantage and attack_auc
                pass

        except Exception as e:
            print(f"   ❌ Privacy attack failed: {e}")
            success = False
            # Keep default values

        # ============================================================
        # FIXED: Ensure attack_advantage is never negative
        # ============================================================
        if attack_advantage < 0:
            print(f"   ⚠️  Attack advantage was negative ({attack_advantage:.4f}), setting to 0")
            attack_advantage = 0.0

        # Calculate privacy risk level based on corrected advantage
        privacy_risk = self._calculate_privacy_risk(attack_advantage)
        risk_score = self._get_privacy_risk_score(attack_advantage)
        confidence_gap = np.mean(train_feats['confidences']) - np.mean(test_feats['confidences'])

        # Plot results (only if successful)
        if success:
            self._plot_attack_distribution(
                train_feats['confidences'], test_feats['confidences'],
                train_feats['entropies'], test_feats['entropies'],
                threshold, attack_advantage, attack_type
            )

        # Print results
        print("\n" + "="*50)
        print("🔐 PRIVACY ATTACK RESULTS")
        print("="*50)
        print(f"Attack Type      : {attack_type.upper()}")
        print(f"Attack Accuracy  : {attack_accuracy:.3f}")
        print(f"Attack AUC       : {attack_auc:.4f}")
        print(f"Attack Advantage : {attack_advantage:.4f}")
        print(f"Confidence Gap   : {confidence_gap:.4f}")
        print(f"Optimal Threshold: {threshold:.3f}")
        print(f"Privacy Risk     : {privacy_risk}")
        print("-"*50)
        
        # Determine privacy level with detailed feedback
        if attack_advantage < 0.02:
            print("🌟 EXCELLENT - Near perfect privacy (random guessing)")
        elif attack_advantage < 0.05:
            print("✅ EXCELLENT - Very low privacy leakage")
        elif attack_advantage < 0.10:
            print("✅ GOOD - Low privacy leakage")
        elif attack_advantage < 0.15:
            print("⚠️ MODERATE - Some privacy leakage detected")
        elif attack_advantage < 0.20:
            print("⚠️ MODERATE - Noticeable privacy leakage")
        else:
            print("❌ HIGH - Significant privacy leakage detected!")
        print("="*50)

        return {
            "attack_advantage": attack_advantage,
            "attack_accuracy": attack_accuracy,
            "attack_auc": attack_auc,
            "threshold": threshold,
            "optimal_threshold": threshold,
            "attack_type": attack_type,
            "train_conf_mean": np.mean(train_feats['confidences']),
            "test_conf_mean": np.mean(test_feats['confidences']),
            "train_entropy_mean": np.mean(train_feats['entropies']),
            "test_entropy_mean": np.mean(test_feats['entropies']),
            "train_loss_mean": np.mean(train_feats['losses']),
            "test_loss_mean": np.mean(test_feats['losses']),
            "train_correctness": np.mean(train_feats['correctness']),
            "test_correctness": np.mean(test_feats['correctness']),
            "privacy_risk": privacy_risk,
            "risk_score": risk_score,
            "confidence_gap": confidence_gap,
            "success": success
        }

    def _get_attack_features(self, loader):
        """
        Extract comprehensive features for membership inference attack.
        Applies temperature scaling if available.
        
        Returns:
            dict: Dictionary containing:
                - confidences: Max softmax probabilities
                - entropies: Prediction entropy
                - losses: Cross-entropy losses
                - predictions: Predicted classes
                - targets: True labels
                - correctness: Whether prediction is correct
                - margins: Difference between top-2 probabilities
        """
        self.model.eval()
        all_confidences = []
        all_entropies = []
        all_losses = []
        all_predictions = []
        all_targets = []
        all_margins = []
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Get temperature from model if available
        temperature = getattr(self.model, 'temperature', 1.0)
        
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions (with temperature scaling if available)
                if hasattr(self.model, 'predict_with_temperature'):
                    # For SISA wrapper
                    probs = self.model.predict_with_temperature(data.cpu().numpy(), temperature)
                    probs = torch.FloatTensor(probs).to(self.device)
                    outputs = None  # Will be computed later if needed
                else:
                    # For standard model
                    outputs = self.model(data)
                    # Apply temperature scaling if attribute exists
                    if hasattr(self.model, 'temperature'):
                        outputs = outputs / self.model.temperature
                    probs = torch.softmax(outputs, dim=1)
                
                # Confidences (max probability)
                max_prob, preds = probs.max(dim=1)
                all_confidences.extend(max_prob.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Margins (difference between top-2 probabilities)
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                all_margins.extend(margins.cpu().numpy())
                
                # Entropies (uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                all_entropies.extend(entropy.cpu().numpy())
                
                # Losses (if outputs available)
                if outputs is not None:
                    losses = criterion(outputs, targets)
                    all_losses.extend(losses.cpu().numpy())
                else:
                    # For SISA wrapper, use cross-entropy with probs
                    losses = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                    all_losses.extend(losses.cpu().numpy())
        
        all_confidences = np.array(all_confidences)
        all_entropies = np.array(all_entropies)
        all_losses = np.array(all_losses)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_margins = np.array(all_margins)
        
        # Print confidence statistics for debugging
        print(f"   Train Conf Mean: {np.mean(all_confidences):.4f}")
        print(f"   Train Entropy Mean: {np.mean(all_entropies):.4f}")
        
        return {
            'confidences': all_confidences,
            'entropies': all_entropies,
            'losses': all_losses,
            'predictions': all_predictions,
            'targets': all_targets,
            'correctness': (all_predictions == all_targets).astype(float),
            'margins': all_margins
        }

    def _confidence_attack(self, train_confs, test_confs):
        """
        Enhanced membership inference based on confidence scores.
        Training samples typically have higher confidence.
        """
        train_labels = np.ones(len(train_confs))
        test_labels = np.zeros(len(test_confs))
        
        all_confs = np.concatenate([train_confs, test_confs])
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Find optimal threshold with better range
        best_threshold = 0.5
        best_acc = 0.5
        best_auc = 0.5
        
        for threshold in np.linspace(0.3, 0.95, 50):
            preds = (all_confs > threshold).astype(int)
            acc = accuracy_score(all_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # Calculate AUC
        try:
            best_auc = roc_auc_score(all_labels, all_confs)
        except:
            best_auc = 0.5
        
        # Calculate Attack Advantage using ROC curve
        try:
            fpr, tpr, _ = roc_curve(all_labels, all_confs)
            attack_advantage = (tpr - fpr).max()
            if attack_advantage < 0:
                attack_advantage = max(0, 2 * (best_acc - 0.5))
        except:
            attack_advantage = max(0, 2 * (best_acc - 0.5))
        
        return best_acc, attack_advantage, best_threshold, best_auc

    def _entropy_attack(self, train_entropy, test_entropy):
        """
        Enhanced membership inference based on entropy.
        Training samples typically have lower entropy (more confident).
        """
        train_labels = np.ones(len(train_entropy))
        test_labels = np.zeros(len(test_entropy))
        
        all_entropy = np.concatenate([train_entropy, test_entropy])
        all_labels = np.concatenate([train_labels, test_labels])
        
        best_threshold = 0.5
        best_acc = 0.5
        best_auc = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 50):
            preds = (all_entropy < threshold).astype(int)
            acc = accuracy_score(all_labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        # Calculate AUC (invert entropy for AUC calculation)
        try:
            inverted_entropy = -all_entropy
            best_auc = roc_auc_score(all_labels, inverted_entropy)
        except:
            best_auc = 0.5
        
        # Calculate Attack Advantage using ROC curve
        try:
            fpr, tpr, _ = roc_curve(all_labels, -all_entropy)
            attack_advantage = (tpr - fpr).max()
            if attack_advantage < 0:
                attack_advantage = max(0, 2 * (best_acc - 0.5))
        except:
            attack_advantage = max(0, 2 * (best_acc - 0.5))
        
        return best_acc, attack_advantage, best_threshold, best_auc

    def _combined_attack(self, train_feats, test_feats):
        """
        Enhanced combined attack using confidence, entropy, margin, and interaction features.
        """
        # Create combined features (7 features)
        train_features = np.column_stack([
            train_feats['confidences'],                          # 1. Confidence
            -train_feats['entropies'],                           # 2. Negative entropy
            train_feats['confidences'] * (1 - train_feats['entropies'] / 2),  # 3. Interaction
            train_feats['correctness'],                          # 4. Correctness
            train_feats['margins'],                              # 5. Margin (top-2 diff)
            train_feats['confidences'] * train_feats['correctness'],  # 6. Conf * Correctness
            -train_feats['losses'],                              # 7. Negative loss
        ])
        
        test_features = np.column_stack([
            test_feats['confidences'],
            -test_feats['entropies'],
            test_feats['confidences'] * (1 - test_feats['entropies'] / 2),
            test_feats['correctness'],
            test_feats['margins'],
            test_feats['confidences'] * test_feats['correctness'],
            -test_feats['losses'],
        ])
        
        train_labels = np.ones(len(train_features))
        test_labels = np.zeros(len(test_features))
        
        all_features = np.concatenate([train_features, test_features], axis=0)
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Normalize features
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
        
        # Train with cross-validation
        X_train, X_val, y_train, y_val = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        
        # Use balanced Random Forest
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Calculate AUC
        y_prob = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        
        # Calculate Attack Advantage using ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            attack_advantage = (tpr - fpr).max()
            if attack_advantage < 0:
                attack_advantage = max(0, 2 * (acc - 0.5))
        except:
            attack_advantage = max(0, 2 * (acc - 0.5))
        
        # Get optimal threshold
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        
        return acc, attack_advantage, threshold, auc

    def _advanced_attack(self, train_feats, test_feats):
        """
        Advanced membership inference attack using multiple features with MLP classifier.
        """
        # Create advanced features (8 features)
        train_features = np.column_stack([
            train_feats['confidences'],                          # 1. Confidence
            -train_feats['entropies'],                           # 2. Negative entropy
            train_feats['losses'],                               # 3. Loss
            train_feats['correctness'],                          # 4. Correctness
            train_feats['confidences'] * (1 - train_feats['entropies'] / 2),  # 5. Interaction
            train_feats['confidences'] / (train_feats['entropies'] + 1e-10),  # 6. Ratio
            train_feats['margins'],                              # 7. Margin
            train_feats['confidences'] * train_feats['margins'], # 8. Conf * Margin
        ])
        
        test_features = np.column_stack([
            test_feats['confidences'],
            -test_feats['entropies'],
            test_feats['losses'],
            test_feats['correctness'],
            test_feats['confidences'] * (1 - test_feats['entropies'] / 2),
            test_feats['confidences'] / (test_feats['entropies'] + 1e-10),
            test_feats['margins'],
            test_feats['confidences'] * test_feats['margins'],
        ])
        
        train_labels = np.ones(len(train_features))
        test_labels = np.zeros(len(test_features))
        
        all_features = np.concatenate([train_features, test_features], axis=0)
        all_labels = np.concatenate([train_labels, test_labels])
        
        # Normalize features
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
        
        # Train with cross-validation
        X_train, X_val, y_train, y_val = train_test_split(
            all_features, all_labels, test_size=0.3, random_state=42, stratify=all_labels
        )
        
        # Use MLP with balanced class weights
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Calculate AUC
        y_prob = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        
        # Calculate Attack Advantage using ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            attack_advantage = (tpr - fpr).max()
            if attack_advantage < 0:
                attack_advantage = max(0, 2 * (acc - 0.5))
        except:
            attack_advantage = max(0, 2 * (acc - 0.5))
        
        # Get optimal threshold
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
        
        print(f"   Advanced Attack AUC: {auc:.3f}")
        print(f"   Cross-validation accuracy: {np.mean(cross_val_score(clf, X_train, y_train, cv=3)):.3f}")
        
        return acc, attack_advantage, threshold, auc

    def _calculate_privacy_risk(self, advantage):
        """
        Calculate overall privacy risk level.
        
        Args:
            advantage: Attack advantage score (TPR - FPR)
            
        Returns:
            str: Risk level
        """
        if advantage < 0.02:
            return "EXCELLENT"
        elif advantage < 0.05:
            return "VERY_GOOD"
        elif advantage < 0.10:
            return "GOOD"
        elif advantage < 0.15:
            return "MODERATE"
        elif advantage < 0.20:
            return "HIGH"
        else:
            return "CRITICAL"

    def _get_privacy_risk_score(self, advantage):
        """
        Convert advantage to numeric risk score (0-100).
        
        Args:
            advantage: Attack advantage
            
        Returns:
            float: Risk score 0-100
        """
        if advantage < 0.02:
            return 5
        elif advantage < 0.05:
            return 15
        elif advantage < 0.10:
            return 30
        elif advantage < 0.15:
            return 50
        elif advantage < 0.20:
            return 75
        else:
            return 95

    def _plot_attack_distribution(self, train_confs, test_confs, 
                                   train_entropy, test_entropy,
                                   threshold, advantage, attack_type):
        """
        Plot attack distribution and save figure.
        Uses non-interactive backend to avoid Tkinter issues.
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Confidence Distribution
            ax1 = axes[0, 0]
            ax1.hist(train_confs, bins=30, alpha=0.6, density=True, 
                     label='Training', color='blue', edgecolor='black')
            ax1.hist(test_confs, bins=30, alpha=0.6, density=True, 
                     label='Testing', color='orange', edgecolor='black')
            ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold = {threshold:.3f}')
            ax1.set_xlabel('Maximum Confidence')
            ax1.set_ylabel('Density')
            ax1.set_title('Confidence Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot 2: Entropy Distribution
            ax2 = axes[0, 1]
            ax2.hist(train_entropy, bins=30, alpha=0.6, density=True,
                     label='Training', color='blue', edgecolor='black')
            ax2.hist(test_entropy, bins=30, alpha=0.6, density=True,
                     label='Testing', color='orange', edgecolor='black')
            ax2.set_xlabel('Entropy (Uncertainty)')
            ax2.set_ylabel('Density')
            ax2.set_title('Entropy Distribution')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Plot 3: Attack Performance Summary
            ax3 = axes[1, 0]
            ax3.text(0.5, 0.85, f'Attack Type: {attack_type.upper()}', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax3.text(0.5, 0.70, f'Attack Advantage: {advantage:.4f}', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            
            # Determine status and color
            if advantage < 0.05:
                color = 'green'
                status = 'EXCELLENT'
                status_text = '✓ Very low privacy leakage'
            elif advantage < 0.10:
                color = 'blue'
                status = 'GOOD'
                status_text = '✓ Low privacy leakage'
            elif advantage < 0.15:
                color = 'orange'
                status = 'MODERATE'
                status_text = '⚠ Some privacy leakage'
            elif advantage < 0.20:
                color = 'darkorange'
                status = 'HIGH'
                status_text = '⚠ Noticeable privacy leakage'
            else:
                color = 'red'
                status = 'CRITICAL'
                status_text = '✗ Significant privacy leakage!'
            
            ax3.text(0.5, 0.55, f'Risk Level: {status}', 
                    ha='center', va='center', fontsize=14, color=color, fontweight='bold')
            ax3.text(0.5, 0.40, status_text, 
                    ha='center', va='center', fontsize=11)
            
            # Add statistics
            conf_gap = np.mean(train_confs) - np.mean(test_confs)
            ax3.text(0.5, 0.20, f'Training Conf: {np.mean(train_confs):.3f} | Test Conf: {np.mean(test_confs):.3f}',
                    ha='center', va='center', fontsize=9, style='italic')
            ax3.text(0.5, 0.08, f'Confidence Gap: {conf_gap:.4f}',
                    ha='center', va='center', fontsize=9, style='italic', 
                    color='red' if conf_gap > 0.05 else 'green')
            
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            
            # Plot 4: ROC Curve Analysis
            ax4 = axes[1, 1]
            
            # Generate ROC curve data
            all_scores = np.concatenate([train_confs, test_confs])
            all_labels = np.concatenate([np.ones(len(train_confs)), 
                                         np.zeros(len(test_confs))])
            
            try:
                fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
                auc = roc_auc_score(all_labels, all_scores)
                ax4.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
                ax4.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC=0.5)')
                ax4.set_xlabel('False Positive Rate (FPR)')
                ax4.set_ylabel('True Positive Rate (TPR)')
                ax4.set_title('ROC Curve Analysis')
                ax4.legend()
                ax4.grid(alpha=0.3)
            except:
                ax4.text(0.5, 0.5, 'ROC Curve unavailable', 
                        ha='center', va='center', fontsize=12)
                ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig('privacy_attack_analysis.png', dpi=200, bbox_inches='tight')
            plt.close('all')
            plt.clf()
            
            print("📊 Privacy attack plot saved to privacy_attack_analysis.png")
            
        except Exception as e:
            print(f"⚠️ Could not save privacy plot: {e}")
            plt.close('all')

    def get_privacy_report(self, train_loader, test_loader):
        """
        Generate comprehensive privacy report with all attack types.
        
        Returns:
            dict: Detailed privacy report
        """
        print("\n" + "="*60)
        print("🔐 PRIVACY REPORT - COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Run all attack types
        attacks = {}
        for attack_type in ['confidence', 'entropy', 'combined', 'advanced']:
            print(f"\n▶ Running {attack_type.upper()} attack...")
            attacks[attack_type] = self.run_attack(train_loader, test_loader, attack_type)
        
        print("\n" + "="*60)
        print("📊 ATTACK RESULTS COMPARISON")
        print("="*60)
        print(f"{'Attack Type':<15} {'Advantage':<12} {'AUC':<10} {'Risk':<12} {'Conf Gap':<10}")
        print("-"*65)
        for name, result in attacks.items():
            conf_gap = result.get('confidence_gap', 0)
            print(f"{name:<15} {result['attack_advantage']:.4f}     {result['attack_auc']:.3f}    {result['privacy_risk']:<12} {conf_gap:.4f}")
        print("="*60)
        
        # Determine most effective attack
        most_effective = max(attacks.items(), key=lambda x: x[1]['attack_advantage'])
        least_effective = min(attacks.items(), key=lambda x: x[1]['attack_advantage'])
        
        print(f"\n📌 Most effective attack: {most_effective[0].upper()}")
        print(f"   Advantage: {most_effective[1]['attack_advantage']:.4f}")
        print(f"   Risk Level: {most_effective[1]['privacy_risk']}")
        
        print(f"\n📌 Least effective attack: {least_effective[0].upper()}")
        print(f"   Advantage: {least_effective[1]['attack_advantage']:.4f}")
        
        # ============================================================
        # FIXED: Calculate average only from valid (non-negative) values
        # ============================================================
        valid_advantages = [r['attack_advantage'] for r in attacks.values() 
                           if r['attack_advantage'] >= 0]
        
        if valid_advantages:
            avg_advantage = np.mean(valid_advantages)
            print(f"\n📊 Average Attack Advantage (from {len(valid_advantages)} valid results): {avg_advantage:.4f}")
        else:
            avg_advantage = 0
            print(f"\n⚠️ No valid attack advantages found")
        
        if avg_advantage < 0.05:
            print("✅ Overall privacy protection: EXCELLENT")
        elif avg_advantage < 0.10:
            print("✅ Overall privacy protection: GOOD")
        elif avg_advantage < 0.15:
            print("⚠️ Overall privacy protection: MODERATE")
        else:
            print("❌ Overall privacy protection: INSUFFICIENT")
        
        return attacks

    def evaluate_defense_effectiveness(self, before_loader, after_loader):
        """
        Evaluate how effective privacy defenses are by comparing before and after.
        
        Args:
            before_loader: DataLoader before applying defenses
            after_loader: DataLoader after applying defenses
            
        Returns:
            dict: Improvement metrics
        """
        print("\n" + "="*60)
        print("📊 DEFENSE EFFECTIVENESS EVALUATION")
        print("="*60)
        
        # Run advanced attacks before and after
        print("\n▶ Evaluating BEFORE defense...")
        before_result = self.run_attack(before_loader, before_loader, 'advanced')
        
        print("\n▶ Evaluating AFTER defense...")
        after_result = self.run_attack(after_loader, after_loader, 'advanced')
        
        # Calculate improvements
        advantage_reduction = before_result['attack_advantage'] - after_result['attack_advantage']
        improvement_pct = (advantage_reduction / (before_result['attack_advantage'] + 1e-10)) * 100
        
        print("\n" + "="*60)
        print("📈 DEFENSE EFFECTIVENESS RESULTS")
        print("="*60)
        print(f"Attack Advantage BEFORE: {before_result['attack_advantage']:.4f}")
        print(f"Attack Advantage AFTER:  {after_result['attack_advantage']:.4f}")
        print(f"Reduction: {advantage_reduction:.4f} ({improvement_pct:.1f}%)")
        print("-"*60)
        
        # Determine effectiveness level
        if improvement_pct > 70:
            status = "🌟 EXCELLENT"
            message = "Your privacy defenses are highly effective!"
        elif improvement_pct > 50:
            status = "✅ GOOD"
            message = "Your privacy defenses are working well."
        elif improvement_pct > 30:
            status = "⚠️ MODERATE"
            message = "Consider applying stronger defenses."
        else:
            status = "❌ LIMITED"
            message = "Your current defenses are not sufficient."
        
        print(f"Status: {status}")
        print(f"Message: {message}")
        print("="*60)
        
        return {
            'before_advantage': before_result['attack_advantage'],
            'after_advantage': after_result['attack_advantage'],
            'reduction': advantage_reduction,
            'improvement_pct': improvement_pct,
            'status': status
        }

    def get_privacy_risk_score(self, attack_results):
        """
        Get numeric privacy risk score.
        
        Args:
            attack_results: Results from run_attack()
            
        Returns:
            float: Privacy risk score (0-100)
        """
        return self._get_privacy_risk_score(attack_results['attack_advantage'])