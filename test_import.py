from src.patient import PatientRecord
from src.parser import FastaParser
from src.alignment import AdvancedSequenceAligner
from src.feature_extraction import AdvancedFeatureExtractor

print("All imports work successfully!")
from src.models import (
    CancerDetectionMLP,
    CancerDetectionCNN1D,
    CancerDetectionTransformer,
)
mlp = CancerDetectionMLP(338)
cnn = CancerDetectionCNN1D(338)
transformer = CancerDetectionTransformer(338)

print("Models imported successfully!")
from src.database import EnhancedDatabase

db = EnhancedDatabase()

print("Database imported successfully!")