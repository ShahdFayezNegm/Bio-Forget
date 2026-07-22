import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
# =================================================================================
# ADVANCED SEQUENCE ALIGNMENT (BLAST-like + Reference Genome)
# =================================================================================

class AdvancedSequenceAligner:
    """
    Advanced alignment supporting:
    1. Reference genome alignment (local/global)
    2. BLAST-like alignment (simulated + optional real BLAST)
    3. Multi-sequence alignment
    """

    def __init__(self, use_real_blast=False):
        self.use_real_blast = use_real_blast
        self.aligner = PairwiseAligner()
        self.aligner.mode = 'local'
       # Set scoring parameters for alignment
        self.aligner.match_score = 2          # Reward for matching bases
        self.aligner.mismatch_score = -1      # Penalty for mismatched bases
        self.aligner.open_gap_score = -5      # Penalty for opening a gap
        self.aligner.extend_gap_score = -2    # Penalty for extending a gap

        # Reference genome sequences for key genes
        self.reference_sequences = {
            'TP53': 'ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGAC',
            'BRCA1': 'ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAGGAGCCTACAAGAAAGTACGAGATTTAGTCAACTTGTTGAAGAGCTATTGAAAATCATTTGTGCTTTTCAGCTTGACACAGGTTTGGAGTATGCAAACAGCTATAATTTTGCAAAAAAGGAAAATAACTCTCCTGAACATCTAAAAGATGAAGTTTCTATCATCCAAAGTATGGGCTACAGAAACCGTGCCAAAAGACTTCTTACAAGTGAACCCGAAAATCCTTCCTTGCA',
            'GAPDH': 'ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTTTACATCTTCCAGTATGACTCCACCCACGGCAAATTCCATGGCACCGTCAAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAATCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTGGTGGAGTCCACTGGCGTCTTCACCACCATGGAGAAGGCTGGGGCTCACTTGCAGGGGGGAGCCAAAAGGGTCATCATCTCTGCCCCCTCTGCTGATGCCCCCATGTTCGTCATGGGTGTGAACCATGAGAAGTATGACAACAGCCTCAAGATCATCAGCAATGCCTCCTGCACCACCAACTGCTTAGCACCCCTGGCCAAGGTCATCCATGACAACTTTGGTATCGTGGAAGGACTCATGACCACAGTCCATGCCATCACTGCCACCCAGAAGACTGTGGATGGCCCCTCCGGGAAACTGTGGCGTGATGGCCGCGGGGCTCTCCAGAACATCATCCCTGCCTCTACTGGCGCTGCCAAGGCTGTGGGCAAGGTCATCCCTGAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTCCCCACTGCCAACGTGTCAGTGACTGACTGCTTCTGCACCTACAATGGCGAGATCAAGGCCCCCTACAAGGACATCAAAGTGGTGAAGCAGGCCTCAGAGGGCACCCTGAAGGGCAACCTGGGCTACACTGAGCACCAGGTGGTCTCCTCTGACTTCAACAGCGACACCCACTCCTCCACCTTTGACGCTGGGGCTGGGATTGCCCTCAACGACCACTTTGTCAAGCTCATTTCTTGGTATGACAACGAATTTGGCTACAGCAACAGGGTGGTGGACCTCATGGCCCACATGGCTCCCAAGGAGTAAG',
            'ACTB': 'ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAGGCCGGCTTCGCGGGCGACGATGCCCCCCGGGCCGTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCCCAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCACCACACCTTCTACAATGAGCTGCGTGTGGCCCCCGAGGAGCACCCCGTGCTGCTGACCGAGGCCCCCCTGAACCCCAAGGCCAACCGCGAGAAGATGACCCAGATCATGTTCGAGACCTTCAACACCCCAGCCATGTACGTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCTGGCCGTACCACTGGCATCGTGATGGACTCCGGTGACGGGGTCACCCACACTGTGCCCATCTACGAGGGGTATGCCCTCCCCCATGCCATCCTGCGTCTGGACCTGGCTGGCCGGGACCTGACTGACTACCTCATGAAGATCCTCACCGAGCGCGGCTACAGCTTCACCACCACGGCCGAGCGGGAAATCGTGCGTGACATTAAGGAGAAGCTGTGCTACGTCGCCCTGGACTTCGAGCAAGAGATGGCCACGGCTGCTTCCAGCTCCTCCCTGGAGAAGAGCTACGAACTGCCTGACGGCCAGGTCATCACCATTGGCAACGAGCGGTTCCGCTGCCCTGAGGCACTCTTCCAGCCTTCCTTCCTGGGCATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGCAAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCCGACAGGATGCAGAAGGAGATCACTGCCCTGGCACCCAGCACAAATGA',
        }

        print(f"[Advanced Aligner] Initialized with {len(self.reference_sequences)} reference sequences")
        print(f"[Advanced Aligner] Real BLAST: {'ENABLED' if use_real_blast else 'DISABLED (using local alignment)'}")
# Find which reference genome gives the best alignment score
    def align_to_reference(self, query_sequence, reference_name):
        """Local alignment to reference genome"""
        if reference_name not in self.reference_sequences:
            return None

        ref_seq = self.reference_sequences[reference_name]
        query_short = query_sequence[:500]
        ref_short = ref_seq[:500]

        alignments = self.aligner.align(query_short, ref_short)

        if alignments:
            best = alignments[0]
            q_aligned = str(best.query)
            r_aligned = str(best.target)
            matches = sum(1 for a, b in zip(q_aligned, r_aligned) if a == b and a != '-')
            total = sum(1 for a, b in zip(q_aligned, r_aligned) if a != '-' or b != '-')
            identity = matches / total if total > 0 else 0

            return {
                'score': best.score,
                'identity': identity,
                'reference': reference_name,
                'alignment_length': len(q_aligned)
            }
        return None

    def find_best_reference(self, query_sequence):
        """Find best matching reference genome"""
        best_result = None
        best_score = -1

        for ref_name in self.reference_sequences.keys():
            result = self.align_to_reference(query_sequence, ref_name)
            if result and result['score'] > best_score:
                best_score = result['score']
                best_result = result

        return best_result

    def run_blast_like_alignment(self, query_sequence, max_hits=5):
        """
        Simulate BLAST-like alignment with multiple hits
        Returns top hits with E-values and bit scores
        """
        results = []

        for ref_name, ref_seq in self.reference_sequences.items():
            align_result = self.align_to_reference(query_sequence, ref_name)
            if align_result:
                # Simulate bit score and E-value
                bit_score = align_result['score'] * 0.5
                e_value = np.exp(-0.1 * align_result['score'])

                results.append({
                    'reference': ref_name,
                    'bit_score': bit_score,
                    'e_value': e_value,
                    'identity': align_result['identity'],
                    'score': align_result['score']
                })

        # Sort by bit score (descending) and return top hits
        results.sort(key=lambda x: x['bit_score'], reverse=True)
        return results[:max_hits]

    def blast_sequence(self, sequence):
        """
        Real BLAST integration (optional - requires internet)
        Falls back to simulated BLAST if real BLAST fails
        """
        if self.use_real_blast:
            try:
                result_handle = NCBIWWW.qblast("blastn", "nt", sequence[:200], hitlist_size=5)
                blast_records = NCBIXML.parse(result_handle)
                results = []
                for blast_record in blast_records:
                    for alignment in blast_record.alignments[:3]:
                        for hsp in alignment.hsps[:1]:
                            results.append({
                                'title': alignment.title,
                                'score': hsp.score,
                                'expect': hsp.expect,
                                'identity': hsp.identities / hsp.align_length
                            })
                return results
            except Exception as e:
                print(f"[BLAST] Failed: {e}, falling back to simulated BLAST")
                return self.run_blast_like_alignment(sequence)
        else:
            return self.run_blast_like_alignment(sequence)

    def extract_alignment_features(self, sequence):
        """Extract comprehensive alignment features for model input"""
        best_match = self.find_best_reference(sequence)
        blast_hits = self.run_blast_like_alignment(sequence)

        features = np.zeros(10)  # Modern alignment feature vector
        if best_match:
            features[0] = best_match['score'] / 1000.0  # Normalized alignment score
            features[1] = best_match['identity']  # Identity percentage
            features[2] = best_match['alignment_length'] / 500.0  # Normalized length

        # Add BLAST-like features
        if blast_hits:
            features[3] = blast_hits[0]['bit_score'] / 100.0 if blast_hits else 0
            features[4] = -np.log10(blast_hits[0]['e_value'] + 1e-10) if blast_hits else 0
            features[5] = len(blast_hits) / 5.0  # Number of significant hits

        # Mutation burden estimate
        if best_match and best_match['identity'] > 0:
            features[6] = 1 - best_match['identity']  # Mutation rate

        return features