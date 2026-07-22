import numpy as np

from collections import Counter
from itertools import product
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.alignment import AdvancedSequenceAligner


class AdvancedFeatureExtractor:
    """
    Feature extractor with:
    1. K-mer features
    2. Genomic composition
    3. Advanced alignment features (BLAST-like)
    """

    def __init__(self, k_values=[3, 4], normalize=True):
        self.k_values = k_values
        self.normalize = normalize
        self.kmer_vocabs = {}
        self.scaler = StandardScaler()
        self.aligner = AdvancedSequenceAligner(use_real_blast=False)

        for k in k_values:
            bases = ['A', 'C', 'G', 'T']
            self.kmer_vocabs[k] = {
                ''.join(kmer): idx
                for idx, kmer in enumerate(product(bases, repeat=k))
            }

        print(f"[Advanced Extractor] K-mer: k={k_values}, Alignment: Advanced (BLAST-like)")

    def _get_feature_count(self):
        total = 0

        for k in self.k_values:
            total += len(self.kmer_vocabs[k])

        total += 8
        total += 10

        return total

    def extract_features(self, sequence):

        # K-mer Features
        all_features = []

        for k in self.k_values:

            kmers = [
                sequence[i:i+k]
                for i in range(len(sequence)-k+1)
                if all(b in "ACGT" for b in sequence[i:i+k])
            ]

            counts = Counter(kmers)

            vec = np.zeros(len(self.kmer_vocabs[k]))

            for kmer, cnt in counts.items():
                if kmer in self.kmer_vocabs[k]:
                    vec[self.kmer_vocabs[k][kmer]] = cnt

            if len(kmers) > 0:
                vec = vec / len(kmers)

            all_features.append(vec)

        combined = np.concatenate(all_features)

        # Genomic Features

        length = len(sequence)

        if length > 0:

            a = sequence.count("A")
            c = sequence.count("C")
            g = sequence.count("G")
            t = sequence.count("T")

            genomic = np.array([
                (g + c) / length,
                (g - c) / (g + c + 1e-6),
                (a - t) / (a + t + 1e-6),
                a / length,
                c / length,
                g / length,
                t / length,
                min(length / 2000, 1)
            ])

            combined = np.concatenate([combined, genomic])

        # Alignment Features

        alignment_features = self.aligner.extract_alignment_features(sequence)

        combined = np.concatenate([
            combined,
            alignment_features
        ])

        # Normalize

        if self.normalize:

            norm = np.linalg.norm(combined)

            if norm > 0:
                combined = combined / norm

        return combined

    def batch_extract(self, sequences):

        features = [
            self.extract_features(seq)
            for seq in tqdm(sequences, desc="Extracting features")
        ]

        features_array = np.array(features)

        if len(features_array) > 1:
            features_array = self.scaler.fit_transform(features_array)

        print(
            f"[Advanced Extractor] Extracted {features_array.shape[1]} features per sample"
        )

        return features_array