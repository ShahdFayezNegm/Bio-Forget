import matplotlib.pyplot as plt
import seaborn as sns
def _plot_attack_distribution(self, train_conf, test_conf, threshold, advantage):
        """Visualize confidence distribution for attack analysis"""
        plt.figure(figsize=(10, 6))
        plt.hist(train_conf, bins=30, alpha=0.5, label='Training Data', color='blue', density=True)
        plt.hist(test_conf, bins=30, alpha=0.5, label='Test Data', color='orange', density=True)
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
        plt.xlabel('Maximum Confidence')
        plt.ylabel('Density')
        plt.title(f'Membership Inference Attack Analysis (Advantage: {advantage:.4f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig('privacy_attack_distribution.png', dpi=150)
        plt.show()
        print(" Privacy attack distribution saved to privacy_attack_distribution.png")


# =================================================================================
# VISUALIZATION FUNCTIONS
# =================================================================================

def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Cancerous'],
                yticklabels=['Healthy', 'Cancerous'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f" Confusion matrix saved to {save_path}")
    plt.show()
