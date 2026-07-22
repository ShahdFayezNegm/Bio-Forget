import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =================================================================================
# DEEP LEARNING MODELS (CNN, Transformer, MLP)
# =================================================================================

# =================================================================================
# 1. Multi-Layer Perceptron (MLP)
# =================================================================================

class CancerDetectionMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for cancer detection from extracted features.
    Simple and interpretable baseline model.
    """
    def __init__(self, input_dim: int, hidden_dims=[128, 64], dropout=0.5):
        super(CancerDetectionMLP, self).__init__()
        layers = []
        
        # Build hidden layers sequentially
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer with 2 classes (cancer vs healthy)
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
        
        print(f"[MLP] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x):
        return self.network(x)


# =================================================================================
# 2. 1D Convolutional Neural Network (CNN1D)
# =================================================================================

class CancerDetectionCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for pattern detection in feature sequences.
    Captures local patterns and spatial dependencies in genomic features.
    """
    def __init__(self, input_dim, n_filters=[64, 128, 256], kernel_sizes=[5, 3, 3], dropout=0.5):
        super(CancerDetectionCNN1D, self).__init__()

        self.input_dim = input_dim
        self.relu = nn.ReLU()
        
        # Convolutional layers with increasing filters
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=n_filters[0], 
            kernel_size=kernel_sizes[0], 
            padding=kernel_sizes[0]//2
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_filters[0], 
            out_channels=n_filters[1], 
            kernel_size=kernel_sizes[1], 
            padding=kernel_sizes[1]//2
        )
        self.conv3 = nn.Conv1d(
            in_channels=n_filters[1], 
            out_channels=n_filters[2], 
            kernel_size=kernel_sizes[2], 
            padding=kernel_sizes[2]//2
        )
        
        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(n_filters[0])
        self.bn2 = nn.BatchNorm1d(n_filters[1])
        self.bn3 = nn.BatchNorm1d(n_filters[2])
        
        # Max pooling to reduce dimensionality
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

        # Calculate output dimension after conv layers
        cnn_output = self._get_cnn_output_dim(input_dim, kernel_sizes)

        # Fully connected layers
        self.fc1 = nn.Linear(cnn_output, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        print(f"[CNN1D] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _get_cnn_output_dim(self, input_dim, kernel_sizes):
        """Calculate output dimension after convolutional and pooling layers."""
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # Add channel dimension (batch, channels, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# =================================================================================
# 3. Improved CNN1D with Residual Connections (Enhanced Privacy)
# =================================================================================

class ImprovedCNN1D(nn.Module):
    """
    Improved CNN1D with residual connections, batch normalization in conv layers,
    and dropout for better accuracy and stability.
    Enhanced with stronger regularization for privacy protection.
    """
    def __init__(self, input_dim, n_filters=[64, 128, 256], dropout=0.5):
        super(ImprovedCNN1D, self).__init__()
        
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        
        # Convolutional blocks with BatchNorm
        self.conv1 = nn.Conv1d(1, n_filters[0], kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(n_filters[0])
        
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters[1])
        
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters[2])
        
        # Residual connections for skip connections
        self.residual1 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size=1)
        self.residual2 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimension after conv layers
        cnn_output = self._get_cnn_output_dim(input_dim)
        
        # Fully connected layers WITHOUT BatchNorm to avoid batch size = 1 issues
        self.fc1 = nn.Linear(cnn_output, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
        print(f"[ImprovedCNN1D] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _get_cnn_output_dim(self, input_dim):
        """Calculate output dimension after convolutional and pooling layers."""
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x)) + self.residual1(x)))
        x = self.pool(self.relu(self.bn3(self.conv3(x)) + self.residual2(x)))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        # Add channel dimension (batch, channels, features)
        x = x.unsqueeze(1)
        
        # First conv block with residual
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Second conv block with residual
        residual = self.residual1(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x)) + residual))
        
        # Third conv block with residual
        residual = self.residual2(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x)) + residual))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layers (no BatchNorm for batch size = 1 support)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# =================================================================================
# 4. Transformer Model
# =================================================================================

class CancerDetectionTransformer(nn.Module):
    """
    Transformer model for capturing long-range dependencies in features.
    Uses self-attention to model relationships between different features.
    """
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.5):
        super(CancerDetectionTransformer, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input features to model dimension
        self.input_projection = nn.Linear(1, d_model)
        
        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.1)

        # Transformer encoder layer with multi-head attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with BatchNorm for stability
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # Keep BatchNorm but ensure batch size > 1 during training
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

        print(f"[Transformer] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x):
        # Add sequence dimension (batch, seq_len, features) where seq_len = input_dim
        x = x.unsqueeze(-1)  # (batch, input_dim, 1)
        x = self.input_projection(x)  # (batch, input_dim, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Classification head
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# =================================================================================
# 5. Lightweight CNN (for faster training)
# =================================================================================

class LightweightCNN1D(nn.Module):
    """
    Lightweight CNN with fewer parameters for faster training.
    Good for quick experimentation and resource-constrained environments.
    """
    def __init__(self, input_dim, dropout=0.3):
        super(LightweightCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Calculate output dimension
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        cnn_output = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(cnn_output, 32)
        self.fc2 = nn.Linear(32, 2)
        
        print(f"[LightweightCNN1D] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# =================================================================================
# 6. Tiny CNN (for very small batch sizes - handles batch size = 1)
# =================================================================================

class TinyCNN1D(nn.Module):
    """
    Tiny CNN for very small datasets and batch sizes.
    No BatchNorm layers to handle batch size = 1.
    """
    def __init__(self, input_dim, dropout=0.2):
        super(TinyCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Calculate output dimension
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        cnn_output = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(cnn_output, 32)
        self.fc2 = nn.Linear(32, 2)
        
        print(f"[TinyCNN1D] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# =================================================================================
# 7. Privacy-Enhanced CNN (with stronger regularization for privacy)
# =================================================================================

class PrivacyEnhancedCNN1D(nn.Module):
    """
    CNN designed for better privacy protection with:
    - Higher dropout (0.5)
    - Weight decay in optimizer
    - Label smoothing friendly
    - Reduced overconfidence
    """
    def __init__(self, input_dim, n_filters=[32, 64, 128], dropout=0.5):
        super(PrivacyEnhancedCNN1D, self).__init__()
        
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, n_filters[0], kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(n_filters[0])
        
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_filters[1])
        
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_filters[2])
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate output dimension
        cnn_output = self._get_cnn_output_dim(input_dim)
        
        # Fully connected layers with strong dropout
        self.fc1 = nn.Linear(cnn_output, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        
        print(f"[PrivacyEnhancedCNN] Created with {sum(p.numel() for p in self.parameters()):,} parameters")

    def _get_cnn_output_dim(self, input_dim):
        """Calculate output dimension after convolutional and pooling layers."""
        x = torch.zeros(1, 1, input_dim)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# =================================================================================
# MODEL FACTORY (for easy model creation)
# =================================================================================

MODEL_REGISTRY = {
    'MLP': CancerDetectionMLP,
    'CNN1D': CancerDetectionCNN1D,
    'ImprovedCNN1D': ImprovedCNN1D,
    'Transformer': CancerDetectionTransformer,
    'LightweightCNN1D': LightweightCNN1D,
    'TinyCNN1D': TinyCNN1D,
    'PrivacyEnhancedCNN': PrivacyEnhancedCNN1D
}

def create_model(model_name, input_dim, **kwargs):
    """
    Factory function to create a model by name.
    
    Args:
        model_name (str): Name of the model
        input_dim (int): Input dimension
        **kwargs: Additional arguments for the model
    
    Returns:
        nn.Module: The created model
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(input_dim=input_dim, **kwargs)


# =================================================================================
# MODEL COMPARISON UTILITY
# =================================================================================

def compare_models(input_dim=338, batch_size=4):
    """
    Compare all models by parameter count and forward pass.
    
    Args:
        input_dim: Input dimension
        batch_size: Batch size for testing
    
    Returns:
        dict: Model comparison results
    """
    results = {}
    
    for model_name, model_class in MODEL_REGISTRY.items():
        model = model_class(input_dim=input_dim)
        params = sum(p.numel() for p in model.parameters())
        
        # Test forward pass
        x = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            output = model(x)
        
        results[model_name] = {
            'parameters': params,
            'output_shape': output.shape,
            'model_class': model_class
        }
    
    return results


# =================================================================================
# TEST MODELS (for verification)
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("TESTING MODELS")
    print("="*60)
    
    input_dim = 338
    batch_sizes = [4, 1]  # Test with both batch sizes
    
    for model_name, model_class in MODEL_REGISTRY.items():
        print(f"\n--- Testing {model_name} ---")
        model = model_class(input_dim=input_dim)
        
        for batch_size in batch_sizes:
            # Create random input
            x = torch.randn(batch_size, input_dim)
            
            # Forward pass
            output = model(x)
            print(f"  Batch size {batch_size}: Input {x.shape} -> Output {output.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)