"""
NCBI Data Loader with Caching System
====================================
This module handles downloading genetic sequences from NCBI with automatic caching.
It includes fallback mechanisms for network failures and data balancing.
"""

from Bio import Entrez, SeqIO
import random
import os
import pickle
import time
from pathlib import Path

# Set your email for NCBI access (required by NCBI)
Entrez.email = "202200678@pua.edu.eg"

# ==============================================================================
# CACHE HELPERS
# ==============================================================================

def get_cache_path():
    """
    Get the path to the cache file.
    
    Returns:
        Path: Path object pointing to the cache file location
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "ncbi_data.pkl"

def save_to_cache(data):
    """
    Save data to cache using pickle serialization.
    
    Args:
        data: The data to be cached (list of sequence dictionaries)
    """
    cache_path = get_cache_path()
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f" Data cached to {cache_path}")

def load_from_cache():
    """
    Load data from cache.
    
    Returns:
        The cached data if exists, None otherwise
    """
    cache_path = get_cache_path()
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f" Data loaded from cache")
        return data
    return None

def is_cache_valid(max_age_hours=24):
    """
    Check if the cache is still valid (not expired).
    
    Args:
        max_age_hours (int): Maximum age of cache in hours
        
    Returns:
        bool: True if cache exists and is valid, False otherwise
    """
    cache_path = get_cache_path()
    if not cache_path.exists():
        return False
    
    # Check file age
    file_age = time.time() - cache_path.stat().st_mtime
    max_age = max_age_hours * 60 * 60
    
    if file_age > max_age:
        print(f" Cache is {file_age/3600:.1f} hours old (max {max_age_hours:.1f} hours)")
        return False
    
    return True

# ==============================================================================
# FETCH FASTA SEQUENCES FROM NCBI
# ==============================================================================

def fetch_fasta(gene, label, max_n=20):
    """
    Download FASTA sequences for a specific human gene from NCBI.

    Args:
        gene (str): Gene name.
        label (int): Class label (1 = Cancer, 0 = Healthy).
        max_n (int): Maximum number of sequences to download.

    Returns:
        list: List of downloaded sequences, each as a dictionary with:
              - id: Sequence identifier
              - sequence: DNA sequence string
              - diagnosis: 1 for cancer, 0 for healthy
              - gene: Gene name
    """

    try:
        # Search for sequences
        handle = Entrez.esearch(
            db="nucleotide",
            term=f"{gene}[Gene] AND Homo sapiens[Organism] AND mRNA",
            retmax=max_n
        )

        ids = Entrez.read(handle)["IdList"]
        handle.close()

        data = []

        # Fetch each sequence
        for seq_id in ids:
            try:
                fetch = Entrez.efetch(
                    db="nucleotide",
                    id=seq_id,
                    rettype="fasta",
                    retmode="text"
                )

                seq_record = SeqIO.read(fetch, "fasta")
                fetch.close()

                data.append({
                    "id": f"{gene}_{seq_record.id}",
                    "sequence": str(seq_record.seq).upper()[:1500],
                    "diagnosis": label,
                    "gene": gene
                })

            except Exception:
                continue

        return data
        
    except Exception as e:
        print(f"    Error fetching {gene}: {e}")
        return []

# ==============================================================================
# GENERATE MOCK DATA (FALLBACK WHEN NCBI FAILS)
# ==============================================================================

def generate_mock_sequence(gene, is_cancer, length=200):
    """
    Generate a mock DNA sequence for testing purposes.
    
    Args:
        gene (str): Gene name (used for identification only)
        is_cancer (bool): If True, add cancer-like mutations
        length (int): Length of the sequence to generate
        
    Returns:
        str: Generated DNA sequence
    """
    bases = ['A', 'T', 'G', 'C']
    seq = ''.join(random.choices(bases, k=length))
    
    if is_cancer:
        # Add cancer-like mutations (more G/C content)
        seq = list(seq)
        for i in range(10, len(seq), 20):
            if random.random() < 0.3:
                seq[i] = random.choice(['G', 'C'])
        seq = ''.join(seq)
    
    return seq

def create_mock_data():
    """
    Create complete mock dataset when NCBI is unavailable.
    
    Returns:
        list: List of mock sequence dictionaries
    """
    print(" Creating mock data for testing...")
    
    cancer_genes = ['TP53', 'BRCA1', 'BRCA2', 'KRAS', 'EGFR', 'BRAF', 'PTEN', 'HER2', 'MYC', 'RB1']
    healthy_genes = ['GAPDH', 'ACTB', 'B2M', 'TBP', 'RPLP0', 'GUSB', 'YWHAZ', 'SDHA', 'RPL13A', 'PPIA']
    
    all_sequences = []
    
    # Create cancer sequences
    for gene in cancer_genes:
        for i in range(20):
            seq = generate_mock_sequence(gene, is_cancer=True)
            all_sequences.append({
                "id": f"{gene}_CANCER_{i:03d}",
                "sequence": seq,
                "diagnosis": 1,
                "gene": gene
            })
    
    # Create healthy sequences
    for gene in healthy_genes:
        for i in range(20):
            seq = generate_mock_sequence(gene, is_cancer=False)
            all_sequences.append({
                "id": f"{gene}_HEALTHY_{i:03d}",
                "sequence": seq,
                "diagnosis": 0,
                "gene": gene
            })
    
    print(f" Created mock data with {len(all_sequences)} sequences")
    return all_sequences

# ==============================================================================
# SAVE SEQUENCES TO FASTA FILE
# ==============================================================================

def save_to_fasta(sequences, filename):
    """
    Save sequences to FASTA format file.
    
    Args:
        sequences (list): List of sequence dictionaries
        filename (str): Output filename
    """
    with open(filename, "w") as f:
        for seq in sequences:
            status = "CANCER" if seq["diagnosis"] == 1 else "HEALTHY"
            f.write(f">{seq['id']}|{status}|{seq['gene']}\n")
            
            # Write sequence in lines of 80 characters (FASTA standard)
            sequence = seq["sequence"]
            for j in range(0, len(sequence), 80):
                f.write(sequence[j:j+80] + "\n")
    
    print(f" Saved to {filename}")

# ==============================================================================
# MAIN DOWNLOAD FUNCTION WITH CACHING
# ==============================================================================

def download_real_ncbi_data(force_download=False, use_mock_if_fail=True):
    """
    Download real NCBI data with automatic caching.
    
    This function handles downloading genetic sequences from NCBI,
    balancing the dataset between cancer and healthy samples,
    and caching the results for future use.
    
    Args:
        force_download (bool): If True, ignore cache and download fresh data
        use_mock_if_fail (bool): If True, create mock data when NCBI fails
    
    Returns:
        str: Path to the generated FASTA file
        
    Example:
        >>> # First time - downloads from NCBI
        >>> fasta_path = download_real_ncbi_data()
        >>> 
        >>> # Second time - uses cache
        >>> fasta_path = download_real_ncbi_data()
        >>> 
        >>> # Force fresh download
        >>> fasta_path = download_real_ncbi_data(force_download=True)
    """
    
    # ======================================================================
    # 1. Check cache first (if not force download)
    # ======================================================================
    if not force_download and is_cache_valid():
        cached_data = load_from_cache()
        if cached_data:
            print(" Using cached data...")
            # Save cache as FASTA file
            fasta_path = "ncbi_data_cached.fasta"
            save_to_fasta(cached_data, fasta_path)
            return fasta_path
    
    # ======================================================================
    # 2. Download from NCBI
    # ======================================================================
    try:
        print("\n" + "=" * 80)
        print(" DOWNLOADING REAL NCBI DATA")
        print("=" * 80)

        # Define gene lists
        cancer_genes = [
            "TP53", "BRCA1", "BRCA2", "KRAS", "EGFR",
            "BRAF", "PTEN", "HER2", "MYC", "RB1"
        ]

        healthy_genes = [
            "GAPDH", "ACTB", "B2M", "TBP", "RPLP0",
            "GUSB", "YWHAZ", "SDHA", "RPL13A", "PPIA"
        ]

        all_sequences = []

        # Download Cancer genes
        print("\n Downloading Cancer genes...")
        for gene in cancer_genes:
            data = fetch_fasta(gene, label=1, max_n=100)
            if data:
                all_sequences.extend(data)
                print(f"    {gene}: {len(data)} sequences")
            else:
                print(f"    {gene}: No sequences found")
                if use_mock_if_fail:
                    print(f"    Using mock data for {gene}")
                    for i in range(10):
                        seq = generate_mock_sequence(gene, is_cancer=True)
                        all_sequences.append({
                            "id": f"{gene}_CANCER_{i:03d}",
                            "sequence": seq,
                            "diagnosis": 1,
                            "gene": gene
                        })

        # Download Healthy genes
        print("\n Downloading Healthy genes...")
        for gene in healthy_genes:
            data = fetch_fasta(gene, label=0, max_n=100)
            if data:
                all_sequences.extend(data)
                print(f"    {gene}: {len(data)} sequences")
            else:
                print(f"    {gene}: No sequences found")
                if use_mock_if_fail:
                    print(f"    Using mock data for {gene}")
                    for i in range(10):
                        seq = generate_mock_sequence(gene, is_cancer=False)
                        all_sequences.append({
                            "id": f"{gene}_HEALTHY_{i:03d}",
                            "sequence": seq,
                            "diagnosis": 0,
                            "gene": gene
                        })

        # Check if we have any data
        if len(all_sequences) == 0:
            print(" No data downloaded!")
            if use_mock_if_fail:
                print(" Creating mock data instead...")
                all_sequences = create_mock_data()
            else:
                return None

        # ======================================================================
        # 3. Balance the dataset (equal number of Cancer and Healthy)
        # ======================================================================
        cancer_seqs = [s for s in all_sequences if s["diagnosis"] == 1]
        healthy_seqs = [s for s in all_sequences if s["diagnosis"] == 0]

        print(f"\n Before balancing: Cancer={len(cancer_seqs)}, Healthy={len(healthy_seqs)}")

        # Take equal samples from both classes
        min_count = min(len(cancer_seqs), len(healthy_seqs))
        
        balanced_cancer = random.sample(cancer_seqs, min_count)
        balanced_healthy = random.sample(healthy_seqs, min_count)

        balanced_sequences = balanced_cancer + balanced_healthy
        random.shuffle(balanced_sequences)

        print(f" After balancing: Cancer={min_count}, Healthy={min_count}")
        print(f" Total: {len(balanced_sequences)} sequences")

        # ======================================================================
        # 4. Save to FASTA file
        # ======================================================================
        output_file = "real_ncbi_data.fasta"
        save_to_fasta(balanced_sequences, output_file)

        # ======================================================================
        # 5. Save to cache for future use
        # ======================================================================
        save_to_cache(balanced_sequences)

        return output_file

    # ======================================================================
    # 6. Handle errors gracefully
    # ======================================================================
    except Exception as e:
        print(f" NCBI download failed: {e}")
        
        if use_mock_if_fail:
            print(" Creating mock data instead...")
            mock_sequences = create_mock_data()
            
            # Save mock data
            output_file = "ncbi_data_mock.fasta"
            save_to_fasta(mock_sequences, output_file)
            
            # Save to cache
            save_to_cache(mock_sequences)
            
            return output_file
        else:
            raise

# ==============================================================================
# UTILITY FUNCTIONS FOR CACHE MANAGEMENT
# ==============================================================================

def get_cache_info():
    """
    Get information about the cache status.
    
    Returns:
        dict: Dictionary containing cache information:
              - exists: Boolean indicating if cache exists
              - size_kb: Size of cache in KB
              - age_hours: Age of cache in hours
              - path: Path to cache file
        
    Example:
        >>> info = get_cache_info()
        >>> if info['exists']:
        >>>     print(f"Cache: {info['size_kb']:.1f} KB, {info['age_hours']:.1f} hours old")
    """
    cache_path = get_cache_path()
    if cache_path.exists():
        size = cache_path.stat().st_size / 1024  # KB
        age = time.time() - cache_path.stat().st_mtime
        return {
            'exists': True,
            'size_kb': size,
            'age_hours': age / 3600,
            'path': str(cache_path)
        }
    else:
        return {
            'exists': False,
            'path': str(cache_path)
        }

def clear_cache():
    """
    Clear the cache by deleting the cache file.
    
    Returns:
        bool: True if cache was cleared, False if no cache existed
        
    Example:
        >>> if clear_cache():
        >>>     print("Cache cleared successfully")
    """
    cache_path = get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
        print(f" Cache cleared: {cache_path}")
        return True
    else:
        print(f" No cache found at {cache_path}")
        return False

# ==============================================================================
# QUICK TEST FUNCTION
# ==============================================================================

def test_data_loader():
    """
    Quick test function to verify the data loader works.
    
    Returns:
        bool: True if test passes, False otherwise
    """
    print(" Testing data loader...")
    
    # Test cache functions
    print(f"Cache path: {get_cache_path()}")
    print(f"Cache valid: {is_cache_valid()}")
    
    # Try downloading data
    fasta_path = download_real_ncbi_data()
    
    if fasta_path and os.path.exists(fasta_path):
        print(f" Test passed! Data saved to: {fasta_path}")
        return True
    else:
        print(" Test failed!")
        return False

# ==============================================================================
# MAIN EXECUTION (for testing)
# ==============================================================================

if __name__ == "__main__":
    # Run test if executed directly
    test_data_loader()