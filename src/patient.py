from dataclasses import dataclass, field
from typing import List, Dict
import hashlib

@dataclass
class PatientRecord:
    patient_id: str
    sequence: str
    diagnosis: int
    gene_name: str = "Unknown"
    hash_id: str = None
    alignment_score: float = 0.0
    alignment_identity: float = 0.0
    blast_results: List[Dict] = field(default_factory=list)

    # Generate hash ID after initialization
    def __post_init__(self):
        if self.hash_id is None:
            self.hash_id = hashlib.sha256(
                f"{self.patient_id}{self.sequence[:100]}".encode()
            ).hexdigest()[:16]