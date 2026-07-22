import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patient_level_split(records, test_size=0.2, random_state=42):
    """
    Split patient records into train/test sets while preserving
    class distribution.
    """
    labels = [r.diagnosis for r in records]

    train_records, test_records = train_test_split(
        records,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels if len(set(labels)) > 1 else None
    )

    return train_records, test_records