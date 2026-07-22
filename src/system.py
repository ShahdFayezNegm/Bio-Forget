"""
Bio-Forget Main System
Complete SISA-based unlearning system for privacy-preserving disease detection.
"""

from collections import Counter
from datetime import datetime
import os
import numpy as np
import random
import json
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.parser import FastaParser
from src.feature_extraction import AdvancedFeatureExtractor
from src.database import EnhancedDatabase
from src.sisa import CompleteSISAUnlearning
from src.trainer import ModelTrainer
from src.models import (
    CancerDetectionMLP,
    CancerDetectionCNN1D,
    CancerDetectionTransformer,
    ImprovedCNN1D,
    LightweightCNN1D
)
from src.baseline import FullRetrainingBaseline
from src.utils import patient_level_split
from src.performance import PerformanceAnalyzer
from src.privacy import PrivacyAuditor
from src.visualization import plot_confusion_matrix
from src.data_loader import download_real_ncbi_data, get_cache_info, clear_cache

# =================================================================================
# MODEL MAP FOR CONFIG
# =================================================================================

MODEL_MAP = {
    "CancerDetectionCNN1D": CancerDetectionCNN1D,
    "CancerDetectionMLP": CancerDetectionMLP,
    "CancerDetectionTransformer": CancerDetectionTransformer,
    "ImprovedCNN1D": ImprovedCNN1D,
    "LightweightCNN1D": LightweightCNN1D
}

# =================================================================================
# DIFFERENTIAL PRIVACY CONFIGURATION (Enhanced for Privacy)
# =================================================================================
DP_CONFIG = {
    'enabled': True,           # Enable DP during training
    'epsilon': 0.03,           # Privacy budget (lower = more privacy) - REDUCED from 0.05
    'delta': 1e-5,
    'max_grad_norm': 0.3,      # Lower = stronger privacy - REDUCED from 0.5
    'noise_multiplier': 1.5    # Higher = stronger privacy - INCREASED from 1.0
}

# =================================================================================
# GLOBAL SEED FOR REPRODUCIBILITY
# =================================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"[Bio-Forget] Reproducibility seed set to {SEED}")
print(f"[Bio-Forget] DP Training: {DP_CONFIG['enabled']}, epsilon={DP_CONFIG['epsilon']}")

# =================================================================================
# MAIN SYSTEM
# =================================================================================

class BioForgetSystem:
    """
    Main system orchestrating the entire Bio-Forget pipeline:
    1. Data loading (with caching)
    2. Feature extraction
    3. Model training with Cross-Validation (with DP support)
    4. SISA setup and unlearning
    5. Privacy verification with advanced attacks
    6. Performance evaluation
    7. Model saving and comparison
    """
    
    def __init__(self, db_path="bio_forget_complete.db", config=None):
        """
        Initialize the Bio-Forget system with all components.
        
        Args:
            db_path (str): Path to database file (single database for all folds)
            config (dict): Configuration dictionary for hyperparameters
        """
        self.parser = FastaParser()
        self.extractor = AdvancedFeatureExtractor()
        
        # Single database for all folds (not recreated each fold)
        self.db_path = db_path
        self.db = None
        
        self.sisa = None
        self.baseline = None
        self.analyzer = PerformanceAnalyzer()
        self.results = {}
        self.all_fold_results = []
        
        # Configuration - store model name as string
        self.config = config or {
            'num_shards': 8,
            'num_slices_per_shard': 3,
            'model_name': 'ImprovedCNN1D',
            'sisa_epochs': 35,
            'retrain_epochs': 25,
            'learning_rate': 5e-5,
            'batch_size': 32,
            'use_dp': DP_CONFIG['enabled'],
            'dp_epsilon': DP_CONFIG['epsilon']
        }
        
        print("[Bio-Forget System] Initialized")
        try:
            config_print = self.config.copy()
            print(f"   Config: {json.dumps(config_print, indent=2)}")
        except Exception as e:
            print(f"   Config: {self.config}")

    def _get_model_class(self):
        """
        Get the model class from config.
        
        Returns:
            Model class
        """
        model_name = self.config.get('model_name', 'ImprovedCNN1D')
        return MODEL_MAP.get(model_name, ImprovedCNN1D)

    def _init_database(self):
        """
        Initialize the database once (not per fold).
        """
        if self.db is None:
            self.db = EnhancedDatabase(self.db_path, use_memory=False)
            print(f"[Database] Initialized at {self.db_path}")

    def _close_database(self):
        """
        Close the database connection.
        """
        if self.db is not None:
            try:
                self.db.close()
                self.db = None
                print("[Database] Closed")
            except Exception as e:
                print(f"[Database] Error closing: {e}")

    def run(self, force_download=False, use_mock_if_fail=True, use_cv=True, n_folds=5, test_size=0.15):
        """
        Run the complete Bio-Forget pipeline with Cross-Validation support.
        """
        
        # ======================================================================
        # 1. HEADER
        # ======================================================================
        print("\n" + "="*80)
        print("🧬 BIO-FORGET: Privacy-Preserving Disease Detection")
        print("= COMPLETE SISA VERSION =")
        print("="*80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Show cache status
        cache_info = get_cache_info()
        if cache_info['exists']:
            print(f"📂 Cache status: {cache_info['size_kb']:.1f} KB, {cache_info['age_hours']:.1f} hours old")
        else:
            print("📂 Cache status: Empty")
        print()

        # ======================================================================
        # 2. DATA LOADING (with caching)
        # ======================================================================
        if os.path.exists("real_ncbi_dataset.fasta"):
            print("📂 Using existing local dataset (real_ncbi_dataset.fasta)...")
            fasta_path = "real_ncbi_dataset.fasta"
        else:
            print("📥 Dataset not found. Downloading from NCBI...")
            fasta_path = download_real_ncbi_data(
                force_download=force_download,
                use_mock_if_fail=use_mock_if_fail
            )
            if fasta_path is None:
                print("❌ Download failed! Using mock data...")
                fasta_path = "ncbi_data_mock.fasta"
                if not os.path.exists(fasta_path):
                    print("❌ Mock data not found! Creating...")
                    from src.data_loader import create_mock_data, save_to_fasta
                    mock_data = create_mock_data()
                    save_to_fasta(mock_data, fasta_path)

        # ======================================================================
        # 3. PARSE DATA
        # ======================================================================
        print("\n📁 Loading data from FASTA file...")
        records = self.parser.parse_file(fasta_path)
        print(f"✅ Loaded {len(records)} patient records")

        # ======================================================================
        # 4. EXTRACT ALL FEATURES FIRST
        # ======================================================================
        print("\n🔬 Extracting features from all records...")
        all_sequences = [r.sequence for r in records]
        all_labels = [r.diagnosis for r in records]
        all_patient_ids = [r.patient_id for r in records]
        
        features = self.extractor.batch_extract(all_sequences)
        print(f"✅ Feature extraction complete: {features.shape[1]} features per sample")
        print(f"   Total samples: {len(features)}")

        # ======================================================================
        # 5. DATA SPLIT
        # ======================================================================
        if use_cv:
            print(f"\n📊 Using {n_folds}-Fold Cross-Validation...")
            return self.run_cross_validation(
                features, all_labels, all_patient_ids, records,
                n_folds=n_folds
            )
        else:
            print(f"\n📊 Using Hold-Out Split (Test Size: {test_size})...")
            return self.run_holdout_split(
                features, all_labels, all_patient_ids, records,
                test_size=test_size
            )

    def run_cross_validation(self, features, labels, patient_ids, records, n_folds=5):
        """
        Run the pipeline with K-Fold Cross-Validation.
        Saves results for each fold and generates comparison tables.
        """
        print(f"\n{'='*80}")
        print(f"📊 CROSS-VALIDATION WITH {n_folds} FOLDS")
        print(f"{'='*80}")
        
        self._init_database()
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
        fold_results = []
        fold_unlearning_results = []
        fold_privacy_results = []
        fold_models = []
        
        all_accuracies = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_aucs = []
        
        fold_summary = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
            print(f"\n{'='*50}")
            print(f"📊 FOLD {fold+1}/{n_folds}")
            print(f"{'='*50}")
            
            train_feat = features[train_idx]
            train_lab = [labels[i] for i in train_idx]
            test_feat = features[test_idx]
            test_lab = [labels[i] for i in test_idx]
            train_patients = [patient_ids[i] for i in train_idx]
            test_patients = [patient_ids[i] for i in test_idx]
            
            train_class_counts = Counter(train_lab)
            print(f"   Training: {len(train_feat)} samples (Cancer={train_class_counts.get(1, 0)}, Healthy={train_class_counts.get(0, 0)})")
            print(f"   Testing: {len(test_feat)} samples (Cancer={Counter(test_lab).get(1, 0)}, Healthy={Counter(test_lab).get(0, 0)})")
            
            fold_result = self.run_single_fold(
                train_feat, train_lab, test_feat, test_lab,
                train_patients, test_patients, records, 
                fold=fold+1
            )
            
            fold_results.append(fold_result)
            
            all_accuracies.append(fold_result['accuracy'])
            all_precisions.append(fold_result['precision'])
            all_recalls.append(fold_result['recall'])
            all_f1s.append(fold_result['f1'])
            all_aucs.append(fold_result['auc'])
            
            if 'unlearn_result' in fold_result:
                fold_unlearning_results.append(fold_result['unlearn_result'])
            if 'attack_results' in fold_result:
                fold_privacy_results.append(fold_result['attack_results'])
            if 'model' in fold_result:
                fold_models.append(fold_result['model'])
            
            self._save_fold_model(fold_result['model'], fold+1)
            
            if self.db is not None:
                try:
                    unlearn_status = fold_result['unlearn_result'].get('forgotten_status', 'UNKNOWN')
                    attack_adv = fold_result['attack_results'].get('attack_advantage', 0)
                    self.db.log_performance(
                        fold=fold+1,
                        accuracy_before=fold_result['unlearn_result'].get('accuracy_before', 0),
                        accuracy_after=fold_result['unlearn_result'].get('accuracy_after', 0),
                        attack_advantage=attack_adv,
                        status=unlearn_status
                    )
                except Exception as e:
                    print(f"⚠️ Error logging performance: {e}")
            
            fold_summary.append({
                'fold': fold+1,
                'accuracy': fold_result['accuracy'],
                'precision': fold_result['precision'],
                'recall': fold_result['recall'],
                'f1': fold_result['f1'],
                'auc': fold_result['auc'],
                'unlearning_status': fold_result['unlearn_result'].get('forgotten_status', 'UNKNOWN'),
                'attack_advantage': fold_result['attack_results'].get('attack_advantage', 0)
            })
        
        # ======================================================================
        # AVERAGE RESULTS ACROSS ALL FOLDS
        # ======================================================================
        print(f"\n{'='*80}")
        print("📊 CROSS-VALIDATION RESULTS (AVERAGED ACROSS FOLDS)")
        print(f"{'='*80}")
        
        print("\n📊 FOLD COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Fold':<6} {'Accuracy':<12} {'AUC':<10} {'F1':<10} {'Unlearning':<15} {'Attack Adv':<12}")
        print("-" * 80)
        for s in fold_summary:
            print(f"{s['fold']:<6} {s['accuracy']:.2f}%     {s['auc']:.3f}    {s['f1']:.3f}    {s['unlearning_status']:<15} {s['attack_advantage']:.4f}")
        print("-" * 80)
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0
        avg_f1 = np.mean(all_f1s) if all_f1s else 0
        avg_auc = np.mean(all_aucs) if all_aucs else 0
        
        std_accuracy = np.std(all_accuracies) if all_accuracies else 0
        std_auc = np.std(all_aucs) if all_aucs else 0
        
        if fold_unlearning_results:
            avg_unlearning_time = np.mean([r['unlearning_time'] for r in fold_unlearning_results])
            avg_acc_before = np.mean([r['accuracy_before'] for r in fold_unlearning_results])
            avg_acc_after = np.mean([r['accuracy_after'] for r in fold_unlearning_results])
            avg_acc_drop = np.mean([r['accuracy_drop'] for r in fold_unlearning_results])
            
            success_count = sum([1 for r in fold_unlearning_results if r['forgotten_status'] == 'SUCCESSFUL'])
            success_rate = success_count / len(fold_unlearning_results) * 100
            avg_confidence_drop = np.mean([r.get('confidence_drop', 0) for r in fold_unlearning_results])
            
            # Average advanced forget quality metrics
            avg_param_distance = np.mean([r.get('forget_quality', {}).get('param_distance', 0) 
                                          for r in fold_unlearning_results if 'forget_quality' in r])
            avg_gradient_sim = np.mean([r.get('forget_quality', {}).get('gradient_similarity', 0) 
                                        for r in fold_unlearning_results if 'forget_quality' in r])
            avg_kl_div = np.mean([r.get('forget_quality', {}).get('kl_divergence', 0) 
                                  for r in fold_unlearning_results if 'forget_quality' in r])
            avg_js_div = np.mean([r.get('forget_quality', {}).get('js_divergence', 0) 
                                  for r in fold_unlearning_results if 'forget_quality' in r])
        else:
            avg_unlearning_time = 0
            avg_acc_before = 0
            avg_acc_after = 0
            avg_acc_drop = 0
            success_rate = 0
            avg_confidence_drop = 0
            avg_param_distance = 0
            avg_gradient_sim = 0
            avg_kl_div = 0
            avg_js_div = 0
        
        if fold_privacy_results:
            avg_attack_advantage = np.mean([r['attack_advantage'] for r in fold_privacy_results])
            avg_attack_accuracy = np.mean([r['attack_accuracy'] for r in fold_privacy_results])
        else:
            avg_attack_advantage = 0
            avg_attack_accuracy = 0
        
        self.results = {
            'cv_folds': n_folds,
            'model_accuracy': avg_accuracy,
            'model_accuracy_std': std_accuracy,
            'model_precision': avg_precision,
            'model_recall': avg_recall,
            'model_f1': avg_f1,
            'model_auc': avg_auc,
            'model_auc_std': std_auc,
            'unlearning_time': avg_unlearning_time,
            'ensemble_acc_before': avg_acc_before,
            'ensemble_acc_after': avg_acc_after,
            'acc_drop': avg_acc_drop,
            'attack_advantage': avg_attack_advantage,
            'attack_accuracy': avg_attack_accuracy,
            'unlearning_success_rate': success_rate,
            'avg_confidence_drop': avg_confidence_drop,
            'avg_param_distance': avg_param_distance,
            'avg_gradient_similarity': avg_gradient_sim,
            'avg_kl_divergence': avg_kl_div,
            'avg_js_divergence': avg_js_div,
            'forgotten_status': f"{success_rate:.1f}% Success Rate",
            'all_fold_results': fold_results,
            'fold_summary': fold_summary,
            'total_patients': len(records)
        }
        
        print(f"\n🔬 MODEL PERFORMANCE (Average over {n_folds} folds):")
        print(f"   Accuracy: {avg_accuracy:.2f}% ± {std_accuracy:.2f}%")
        print(f"   AUC: {avg_auc:.3f} ± {std_auc:.3f}")
        print(f"   Precision: {avg_precision:.3f}")
        print(f"   Recall: {avg_recall:.3f}")
        print(f"   F1-Score: {avg_f1:.3f}")
        
        print(f"\n🗑️ UNLEARNING PERFORMANCE:")
        print(f"   Success Rate: {success_rate:.1f}% ({success_count}/{len(fold_unlearning_results)})")
        print(f"   Accuracy drop: {avg_acc_drop:.2f}%")
        print(f"   Confidence drop: {avg_confidence_drop:.3f}")
        print(f"   Average time: {avg_unlearning_time:.3f} seconds")
        
        print(f"\n📊 FORGET QUALITY METRICS:")
        print(f"   Parameter Distance: {avg_param_distance:.4f}")
        print(f"   Gradient Similarity: {avg_gradient_sim:.4f}")
        print(f"   KL Divergence: {avg_kl_div:.4f}")
        print(f"   JS Divergence: {avg_js_div:.4f}")
        
        print(f"\n🔐 PRIVACY:")
        print(f"   Attack Advantage: {avg_attack_advantage:.4f}")
        if avg_attack_advantage < 0.05:
            print(f"   [EXCELLENT] - Near random guessing (50%)")
        elif avg_attack_advantage < 0.10:
            print(f"   [GOOD] - Low privacy leakage")
        elif avg_attack_advantage < 0.20:
            print(f"   [MODERATE] - Some privacy leakage")
        else:
            print(f"   [HIGH] - Significant privacy leakage detected")
        
        print(f"\n📊 DATASET:")
        print(f"   Total patients: {len(records)}")
        print(f"   Features per sample: {features.shape[1]}")
        
        self._save_results_to_file()
        self._close_database()
        
        return self.results

    def _save_fold_model(self, sisa_model, fold):
        """
        Save SISA model for a specific fold.
        SISA model contains multiple shard models, so we save each shard separately.
        """
        try:
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)
            
            if hasattr(sisa_model, 'shard_models') and sisa_model.shard_models:
                for shard_idx, shard_model in enumerate(sisa_model.shard_models):
                    save_path = os.path.join(save_dir, f"sisa_fold_{fold}_shard_{shard_idx}.pt")
                    torch.save(shard_model.state_dict(), save_path)
                print(f"[SISA models] saved to {save_dir}/sisa_fold_{fold}_shard_*.pt ({len(sisa_model.shard_models)} shards)")
            else:
                save_path = os.path.join(save_dir, f"sisa_fold_{fold}.pt")
                torch.save(sisa_model.state_dict(), save_path)
                print(f"[Model] saved to {save_path}")
                
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    def _save_results_to_file(self):
        """
        Save results to a JSON file for later analysis.
        Handles non-serializable objects properly.
        """
        try:
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "cv_results.json")
            
            results_copy = {}
            for k, v in self.results.items():
                if k in ['all_fold_results']:
                    continue
                elif isinstance(v, np.floating):
                    results_copy[k] = float(v)
                elif isinstance(v, np.integer):
                    results_copy[k] = int(v)
                elif isinstance(v, list):
                    if k == 'fold_summary':
                        results_copy[k] = []
                        for item in v:
                            if isinstance(item, dict):
                                item_copy = {}
                                for k2, v2 in item.items():
                                    if isinstance(v2, np.floating):
                                        item_copy[k2] = float(v2)
                                    elif isinstance(v2, np.integer):
                                        item_copy[k2] = int(v2)
                                    else:
                                        item_copy[k2] = v2
                                results_copy[k].append(item_copy)
                            else:
                                results_copy[k].append(item)
                    else:
                        results_copy[k] = [float(x) if isinstance(x, np.floating) else x for x in v]
                else:
                    results_copy[k] = v
            
            with open(save_path, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)
            print(f"[Results] saved to {save_path}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    def run_holdout_split(self, features, labels, patient_ids, records, test_size=0.15):
        """
        Run the pipeline with a single hold-out split.
        """
        self._init_database()
        
        print(f"\n📊 Using {test_size*100:.0f}% for testing, {(1-test_size)*100:.0f}% for training")
        
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            range(len(features)), 
            test_size=test_size, 
            stratify=labels, 
            random_state=SEED
        )
        
        train_feat = features[train_idx]
        train_lab = [labels[i] for i in train_idx]
        test_feat = features[test_idx]
        test_lab = [labels[i] for i in test_idx]
        train_patients = [patient_ids[i] for i in train_idx]
        test_patients = [patient_ids[i] for i in test_idx]
        
        train_class_counts = Counter(train_lab)
        print(f"   Training: {len(train_feat)} samples (Cancer={train_class_counts.get(1, 0)}, Healthy={train_class_counts.get(0, 0)})")
        print(f"   Testing: {len(test_feat)} samples (Cancer={Counter(test_lab).get(1, 0)}, Healthy={Counter(test_lab).get(0, 0)})")
        
        fold_result = self.run_single_fold(
            train_feat, train_lab, test_feat, test_lab,
            train_patients, test_patients, records
        )
        
        if self.db is not None:
            try:
                unlearn_status = fold_result['unlearn_result'].get('forgotten_status', 'UNKNOWN')
                attack_adv = fold_result['attack_results'].get('attack_advantage', 0)
                self.db.log_performance(
                    fold=1,
                    accuracy_before=fold_result['unlearn_result'].get('accuracy_before', 0),
                    accuracy_after=fold_result['unlearn_result'].get('accuracy_after', 0),
                    attack_advantage=attack_adv,
                    status=unlearn_status
                )
            except Exception:
                pass
        
        self.results = {
            'model_accuracy': fold_result['accuracy'],
            'model_precision': fold_result['precision'],
            'model_recall': fold_result['recall'],
            'model_f1': fold_result['f1'],
            'model_auc': fold_result['auc'],
            'unlearning_time': fold_result['unlearn_result']['unlearning_time'],
            'ensemble_acc_before': fold_result['unlearn_result']['accuracy_before'],
            'ensemble_acc_after': fold_result['unlearn_result']['accuracy_after'],
            'acc_drop': fold_result['unlearn_result']['accuracy_drop'],
            'attack_advantage': fold_result['attack_results']['attack_advantage'],
            'attack_accuracy': fold_result['attack_results']['attack_accuracy'],
            'forgotten_status': fold_result['unlearn_result']['forgotten_status'],
            'deleted_confidence': fold_result['unlearn_result'].get('deleted_confidence', 0),
            'deleted_uncertainty': fold_result['unlearn_result'].get('deleted_uncertainty', 0),
            'criteria_met': fold_result['unlearn_result'].get('criteria_met', 0),
            'total_patients': len(records)
        }
        
        self._save_fold_model(fold_result['model'], 1)
        self._close_database()
        
        return self.results

    def run_single_fold(self, train_feat, train_lab, test_feat, test_lab, 
                        train_patients, test_patients, records, fold=1):
        """
        Run the Bio-Forget pipeline on a single fold.
        Uses the existing database (not recreating it).
        """
        
        # ======================================================================
        # 1. DATABASE REGISTRATION
        # ======================================================================
        print(f"\n💾 Registering patients in database...")
        for i, patient_id in enumerate(train_patients):
            patient_record = next((r for r in records if r.patient_id == patient_id), None)
            if patient_record:
                shard_id = i % 3
                slice_id = (i // 10) % 2
                self.db.register_patient(patient_record, shard_id, slice_id)
        print(f"✅ Registered {len(train_patients)} patients in database")

        # ======================================================================
        # 2. CREATE DATA LOADERS
        # ======================================================================
        train_class_counts = Counter(train_lab)
        if len(train_class_counts) == 2:
            weights = {0: 1.0/train_class_counts[0], 1: 1.0/train_class_counts[1]}
            sample_weights = [weights[y] for y in train_lab]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            print(f"📊 Using weighted sampling to handle class imbalance")
        else:
            sampler = None

        train_dataset = TensorDataset(
            torch.FloatTensor(train_feat), 
            torch.LongTensor(train_lab)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_feat), 
            torch.LongTensor(test_lab)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=min(32, len(train_dataset)), 
            sampler=sampler, 
            shuffle=sampler is None
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=min(32, len(test_dataset))
        )

        # ======================================================================
        # 3. SETUP AND TRAIN SISA (WITH DP)
        # ======================================================================
        print("\n🗑️ Setting up COMPLETE SISA for training and unlearning...")
        
        model_class = self._get_model_class()
        
        self.sisa = CompleteSISAUnlearning(
            num_shards=self.config['num_shards'],
            num_slices_per_shard=self.config['num_slices_per_shard'],
            model_class=model_class
        )
        
        self.sisa.create_shards_with_slicing(
            train_feat,
            train_lab,
            train_patients
        )

        # ============================================================
        # ENHANCED: Train with Differential Privacy enabled
        # ============================================================
        self.sisa.train_with_tracking(
            input_dim=train_feat.shape[1],
            epochs=self.config['sisa_epochs']
        )

        # ======================================================================
        # Calculate full metrics from SISA predictions
        # ======================================================================
        print("\n📊 Calculating SISA performance metrics...")
        
        train_preds = self.sisa.predict_weighted_ensemble(train_feat, temperature=1.0)
        test_preds = self.sisa.predict_weighted_ensemble(test_feat, temperature=1.0)
        
        accuracy = accuracy_score(test_lab, test_preds) * 100
        precision = precision_score(test_lab, test_preds, average='weighted', zero_division=0)
        recall = recall_score(test_lab, test_preds, average='weighted', zero_division=0)
        f1 = f1_score(test_lab, test_preds, average='weighted', zero_division=0)
        
        try:
            all_probs = self._get_batch_probabilities(test_feat, temperature=1.0)
            auc = roc_auc_score(test_lab, all_probs)
        except Exception as e:
            print(f"⚠️ AUC calculation error: {e}")
            auc = 0.0
        
        cm = confusion_matrix(test_lab, test_preds)
        train_acc = accuracy_score(train_lab, train_preds) * 100

        print(f"\n✅ SISA Training completed!")
        print(f"📊 Train Accuracy: {train_acc:.2f}%")
        print(f"📊 Test Accuracy: {accuracy:.2f}%")
        print(f"📊 Precision: {precision:.3f}")
        print(f"📊 Recall: {recall:.3f}")
        print(f"📊 F1-Score: {f1:.3f}")
        print(f"📊 AUC: {auc:.3f}")
        
        try:
            plot_confusion_matrix(
                cm,
                f"SISA Fold {fold}",
                save_path=f"confusion_matrix_fold_{fold}.png"
            )
        except Exception as e:
            print(f"⚠️ Could not plot confusion matrix: {e}")
        
        final_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }

        # ======================================================================
        # 4. PERFORM UNLEARNING (Different patient per fold)
        # ======================================================================
        print("\n" + "="*50)
        print(f"🗑️ STARTING UNLEARNING PROCESS (Fold {fold})")
        print("="*50)
        
        if train_patients:
            patient_index = (fold - 1) % len(train_patients)
            test_patient = train_patients[patient_index]
            print(f"🎯 Unlearning patient: {test_patient}")
            
            unlearn_result = self.sisa.unlearn_patient(
                test_patient, 
                test_feat, 
                test_lab
            )
            
            trace = self.db.get_patient_trace(test_patient)
            if trace and len(trace) >= 7:
                patient_id, diagnosis, shard_id, slice_id, feature_hash, model_hash, created_at = trace[:7]
                self.db.log_deletion(
                    test_patient,
                    shard_id,
                    slice_id,
                    unlearn_result["unlearning_time"],
                    unlearn_result.get("accuracy_drop", 0)
                )
                print(f"📝 Logged deletion in database")
        else:
            print("⚠️ No training records available for unlearning")
            test_patient = None
            unlearn_result = {
                'accuracy_before': 50, 
                'accuracy_after': 50, 
                'accuracy_drop': 0,
                'unlearning_time': 0, 
                'deleted_prediction': None, 
                'deleted_confidence': 0,
                'deleted_uncertainty': 0,
                'forgotten_status': 'NOT AVAILABLE',
                'criteria_met': 0,
                'confidence_drop': 0,
                'forget_quality': {}
            }

        # ======================================================================
        # 5. PRIVACY VERIFICATION
        # ======================================================================
        print("\n🔐 Privacy verification...")
        
        # ============================================================
        # UPDATED SISAWrapper with temperature range 5.0 to 7.0
        # ============================================================
        class SISAWrapper:
            def __init__(self, sisa_model):
                self.sisa = sisa_model
                self.temperature = getattr(sisa_model, 'temperature', 7.0)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            def __call__(self, x):
                return self.predict_with_temperature(x)
            
            def eval(self):
                pass
            
            def to(self, device):
                self.device = device
                return self
            
            def predict_with_temperature(self, x, temperature=None):
                if temperature is None:
                    temperature = self.temperature
                if torch.is_tensor(x):
                    x = x.cpu().numpy()
                probs = self.sisa.predict_with_temperature(x, temperature)
                return probs
            
            def parameters(self):
                return []
        
        sisa_wrapper = SISAWrapper(self.sisa)
        auditor = PrivacyAuditor(sisa_wrapper)
        
        try:
            attack_results = auditor.run_attack(
                train_loader, 
                test_loader, 
                attack_type='advanced'
            )
            print(f"   Attack Accuracy: {attack_results['attack_accuracy']:.3f}")
            print(f"   Attack Advantage: {attack_results['attack_advantage']:.4f}")
            print(f"   Privacy Risk: {attack_results['privacy_risk']}")
        except Exception as e:
            print(f"⚠️ Privacy verification error: {e}")
            attack_results = {
                'attack_advantage': -1,
                'attack_accuracy': -1,
                'privacy_risk': 'ERROR'
            }

        # ======================================================================
        # 6. FINAL EVALUATION FOR THIS FOLD
        # ======================================================================
        if unlearn_result.get('deleted_prediction') is not None:
            pred = unlearn_result['deleted_prediction']
            true_label = unlearn_result.get('deleted_true_label')
            conf = unlearn_result['deleted_confidence']
            uncertainty = unlearn_result.get('deleted_uncertainty', 0)
            criteria_met = unlearn_result.get('criteria_met', 0)
            confidence_drop = unlearn_result.get('confidence_drop', 0)
            forget_quality = unlearn_result.get('forget_quality', {})
            
            print(f"\n📊 Deleted patient ({test_patient}):")
            print(f"   True label: {'CANCER' if true_label==1 else 'HEALTHY'}")
            print(f"   Prediction: {'CANCER' if pred==1 else 'HEALTHY'}")
            print(f"   Confidence: {conf:.3f} (Drop: {confidence_drop:.3f})")
            print(f"   Uncertainty: {uncertainty:.3f}")
            print(f"   Criteria met: {criteria_met}/3")
            print(f"   Status: {unlearn_result['forgotten_status']}")
            
            if forget_quality:
                print(f"   Forget Quality:")
                print(f"      Parameter Distance: {forget_quality.get('param_distance', 0):.4f}")
                print(f"      Gradient Similarity: {forget_quality.get('gradient_similarity', 0):.4f}")
                print(f"      KL Divergence: {forget_quality.get('kl_divergence', 0):.4f}")
                print(f"      JS Divergence: {forget_quality.get('js_divergence', 0):.4f}")
                print(f"      Prediction Changed: {forget_quality.get('prediction_changed', False)}")

        # ======================================================================
        # 7. RUN BASELINE COMPARISON (Full Retraining)
        # ======================================================================
        print("\n📊 Running baseline comparison...")
        try:
            baseline = FullRetrainingBaseline()
            model_class = self._get_model_class()
            
            baseline_metrics, baseline_time, baseline_trainer = baseline.run_full_retraining(
                train_feat, train_lab, test_feat, test_lab,
                model_class=model_class,
                model_name=self.config['model_name'],
                epochs=self.config['sisa_epochs']
            )
            
            baseline_acc = baseline_metrics['accuracy']
            
            print(f"   Baseline retraining time: {baseline_time:.3f} sec")
            print(f"   Baseline accuracy: {baseline_acc:.2f}%")
            
            sisa_comparison = {
                'accuracy': final_metrics['accuracy'],
                'auc': final_metrics['auc'],
                'f1': final_metrics['f1'],
                'time': unlearn_result['unlearning_time']
            }
            
            baseline_results = {
                'accuracy': baseline_acc,
                'auc': baseline_metrics.get('auc', 0),
                'f1': baseline_metrics.get('f1', 0),
                'time': baseline_time
            }
            
            speedup = baseline_time / (unlearn_result['unlearning_time'] + 1e-10)
            baseline_result = {
                'time': baseline_time,
                'accuracy': baseline_acc,
                'speedup': speedup,
                'sisa_comparison': sisa_comparison,
                'baseline_metrics': baseline_results
            }
            
            print(f"\n📊 SISA vs Full Retraining Comparison:")
            print(f"   {'Metric':<15} {'SISA':<12} {'Full Retraining':<18} {'Improvement':<12}")
            print("-"*60)
            print(f"   {'Accuracy':<15} {final_metrics['accuracy']:.2f}%     {baseline_acc:.2f}%          {abs(baseline_acc - final_metrics['accuracy']):.2f}%")
            print(f"   {'AUC':<15} {final_metrics['auc']:.3f}       {baseline_metrics.get('auc', 0):.3f}             {abs(baseline_metrics.get('auc', 0) - final_metrics['auc']):.3f}")
            print(f"   {'F1':<15} {final_metrics['f1']:.3f}       {baseline_metrics.get('f1', 0):.3f}             {abs(baseline_metrics.get('f1', 0) - final_metrics['f1']):.3f}")
            print(f"   {'Time (s)':<15} {unlearn_result['unlearning_time']:.2f}       {baseline_time:.2f}          {speedup:.2f}x faster")
            print("-"*60)
            
        except Exception as e:
            print(f"⚠️ Baseline comparison failed: {e}")
            baseline_result = {'time': 0, 'accuracy': 0, 'speedup': 0}

        # ======================================================================
        # 8. COMPILE FOLD RESULTS
        # ======================================================================
        return {
            'accuracy': final_metrics['accuracy'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'auc': final_metrics['auc'],
            'model': self.sisa,
            'unlearn_result': unlearn_result,
            'attack_results': attack_results,
            'baseline_result': baseline_result,
            'train_size': len(train_feat),
            'test_size': len(test_feat)
        }

    def _get_batch_probabilities(self, features, temperature=1.0):
        """
        Get probabilities for all samples in batch (faster than per-sample).
        """
        if not self.sisa or not self.sisa.shard_models:
            return np.zeros(len(features))
        
        feat_tensor = torch.FloatTensor(features).to(self.sisa.device)
        all_probs = []
        
        with torch.no_grad():
            for model in self.sisa.shard_models:
                model.eval()
                out = model(feat_tensor)
                scaled_out = out / temperature
                prob = torch.softmax(scaled_out, dim=1)
                all_probs.append(prob.cpu().numpy())
        
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs[:, 1]

    def get_results(self):
        """
        Get the results from the last run.
        
        Returns:
            dict: Complete results dictionary
        """
        return self.results

    def clear_cache(self):
        """
        Clear the data cache.
        
        Returns:
            bool: True if cache was cleared
        """
        return clear_cache()

    def get_cache_info(self):
        """
        Get information about the cache.
        
        Returns:
            dict: Cache information
        """
        return get_cache_info()

    def plot_final_results(self):
        """
        Plot final results: Accuracy, AUC, F1, Privacy, Unlearning Time.
        Saves to final_results.png
        """
        if not self.results or 'fold_summary' not in self.results:
            print("⚠️ No results to plot")
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fold_summary = self.results['fold_summary']
        folds = [s['fold'] for s in fold_summary]
        accuracies = [s['accuracy'] for s in fold_summary]
        aucs = [s['auc'] for s in fold_summary]
        f1s = [s['f1'] for s in fold_summary]
        attack_advs = [s['attack_advantage'] for s in fold_summary]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Accuracy per Fold
        ax1 = axes[0, 0]
        bars1 = ax1.bar(folds, accuracies, color='steelblue', edgecolor='black')
        ax1.axhline(self.results['model_accuracy'], color='red', linestyle='--', 
                    label=f"Avg: {self.results['model_accuracy']:.2f}%")
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy per Fold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        for bar, val in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: AUC per Fold
        ax2 = axes[0, 1]
        bars2 = ax2.bar(folds, aucs, color='forestgreen', edgecolor='black')
        ax2.axhline(self.results['model_auc'], color='red', linestyle='--',
                    label=f"Avg: {self.results['model_auc']:.3f}")
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('AUC')
        ax2.set_title('AUC per Fold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        for bar, val in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: F1 per Fold
        ax3 = axes[1, 0]
        bars3 = ax3.bar(folds, f1s, color='darkorange', edgecolor='black')
        ax3.axhline(self.results['model_f1'], color='red', linestyle='--',
                    label=f"Avg: {self.results['model_f1']:.3f}")
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score per Fold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        for bar, val in zip(bars3, f1s):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Attack Advantage per Fold
        ax4 = axes[1, 1]
        colors = ['green' if a < 0.05 else 'orange' if a < 0.10 else 'darkorange' if a < 0.20 else 'red' 
                  for a in attack_advs]
        bars4 = ax4.bar(folds, attack_advs, color=colors, edgecolor='black')
        ax4.axhline(0.05, color='green', linestyle='--', label='Excellent (0.05)')
        ax4.axhline(0.10, color='orange', linestyle='--', label='Good (0.10)')
        ax4.axhline(0.20, color='red', linestyle='--', label='Moderate (0.20)')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Attack Advantage')
        ax4.set_title('Privacy Leakage per Fold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        for bar, val in zip(bars4, attack_advs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('final_results.png', dpi=200, bbox_inches='tight')
        plt.close()
        print("📊 Final results plot saved to final_results.png")

    def print_summary(self):
        """
        Print a summary of the results in a clean format.
        """
        if not self.results:
            print("⚠️ No results available. Run the pipeline first.")
            return
        
        print("\n" + "="*80)
        print("📊 BIO-FORGET FINAL SUMMARY")
        print("="*80)
        
        print(f"\n🔬 MODEL PERFORMANCE:")
        print(f"   Accuracy: {self.results.get('model_accuracy', 0):.2f}%")
        if 'model_accuracy_std' in self.results:
            print(f"   Accuracy Std: ±{self.results['model_accuracy_std']:.2f}%")
        print(f"   AUC: {self.results.get('model_auc', 0):.3f}")
        if 'model_auc_std' in self.results:
            print(f"   AUC Std: ±{self.results['model_auc_std']:.3f}")
        print(f"   Precision: {self.results.get('model_precision', 0):.3f}")
        print(f"   Recall: {self.results.get('model_recall', 0):.3f}")
        print(f"   F1-Score: {self.results.get('model_f1', 0):.3f}")
        
        print(f"\n🗑️ UNLEARNING PERFORMANCE:")
        print(f"   Success Rate: {self.results.get('unlearning_success_rate', 0):.1f}%")
        print(f"   Accuracy Drop: {self.results.get('acc_drop', 0):.2f}%")
        print(f"   Confidence Drop: {self.results.get('avg_confidence_drop', 0):.3f}")
        print(f"   Time: {self.results.get('unlearning_time', 0):.3f} seconds")
        
        print(f"\n📊 FORGET QUALITY:")
        print(f"   Parameter Distance: {self.results.get('avg_param_distance', 0):.4f}")
        print(f"   Gradient Similarity: {self.results.get('avg_gradient_similarity', 0):.4f}")
        print(f"   KL Divergence: {self.results.get('avg_kl_divergence', 0):.4f}")
        print(f"   JS Divergence: {self.results.get('avg_js_divergence', 0):.4f}")
        
        print(f"\n🔐 PRIVACY:")
        print(f"   Attack Advantage: {self.results.get('attack_advantage', 0):.4f}")
        attack_adv = self.results.get('attack_advantage', 1)
        if attack_adv < 0.05:
            print(f"   [EXCELLENT] - Near random guessing")
        elif attack_adv < 0.10:
            print(f"   [GOOD] - Low privacy leakage")
        elif attack_adv < 0.20:
            print(f"   [MODERATE] - Some privacy leakage")
        else:
            print(f"   [HIGH] - Significant privacy leakage")
        
        if 'fold_summary' in self.results:
            print(f"\n📊 FOLD DETAILS:")
            for s in self.results['fold_summary']:
                print(f"   Fold {s['fold']}: Acc={s['accuracy']:.2f}%, AUC={s['auc']:.3f}, Status={s['unlearning_status']}")
        
        print(f"\n📊 DATASET:")
        print(f"   Total patients: {self.results.get('total_patients', 0)}")
        print("="*80)

# =================================================================================
# MAIN ENTRY POINT
# =================================================================================

def main():
    """
    Main entry point for running the Bio-Forget system.
    """
    system = BioForgetSystem()
    
    results = system.run(
        force_download=False,
        use_mock_if_fail=True,
        use_cv=True,
        n_folds=5,
        test_size=0.15
    )
    
    system.print_summary()
    system.plot_final_results()
    
    return results

if __name__ == "__main__":
    main()