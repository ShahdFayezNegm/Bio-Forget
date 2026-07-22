"""
Enhanced Database Module with Full Traceability for SISA Unlearning
Supports patient tracking, shard mapping, and deletion logging with proper connection management.
"""

import os
import sqlite3
import hashlib
from datetime import datetime

# =================================================================================
# ENHANCED DATABASE WITH FULL TRACEABILITY (FIXED)
# =================================================================================

class EnhancedDatabase:
    """
    Enhanced database with full traceability for SISA unlearning.
    Handles patient registration, shard tracking, deletion logging,
    and feature contribution tracking with proper connection management.
    """
    
    def __init__(self, db_path="bio_forget_complete.db", use_memory=False):
        """
        Initialize database connection.
        
        Args:
            db_path (str): Path to database file
            use_memory (bool): If True, use in-memory database (faster for CV)
        """
        self.db_path = ":memory:" if use_memory else db_path
        self.use_memory = use_memory
        
        # Reset database file if exists (for clean start)
        if not use_memory and os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f" Removed existing database: {db_path}")
            except Exception as e:
                print(f" Could not remove existing database: {e}")
        
        # Create connection with proper settings
        self.conn = None
        self.cursor = None
        self.is_open = False
        self._connect()
        self._create_tables()
        print("[Enhanced Database] Initialized with full traceability schema")
    
    def _connect(self):
        """
        Establish database connection with proper settings for concurrent access.
        """
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0  # Increased timeout for concurrent access
            )
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self.is_open = True
        except sqlite3.Error as e:
            print(f" Database connection error: {e}")
            # Fallback to in-memory database if file access fails
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            self.is_open = True
            self.db_path = ":memory:"
            self.use_memory = True
    
    def _ensure_connection(self):
        """
        Ensure database connection is open, reconnect if needed.
        """
        if not self.is_open or self.conn is None:
            self._connect()
    
    def _create_tables(self):
        """
        Create all necessary tables with full traceability schema.
        """
        self._ensure_connection()
        
        try:
            # Patients table with full tracing
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                diagnosis INTEGER,
                shard_id INTEGER,
                slice_id INTEGER,
                feature_hash TEXT,
                model_parameters_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # Shard tracking table
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS shards (
                shard_id INTEGER PRIMARY KEY,
                num_samples INTEGER,
                slice_count INTEGER,
                model_path TEXT,
                accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            # Deletion log with traceability
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS deletion_log (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                original_shard_id INTEGER,
                original_slice_id INTEGER,
                time_taken REAL,
                accuracy_drop REAL,
                deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )''')

            # Feature contribution tracking
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS feature_contributions (
                contribution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                feature_index INTEGER,
                contribution_weight REAL,
                model_layer TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )''')

            # Performance metrics table
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fold INTEGER,
                accuracy_before REAL,
                accuracy_after REAL,
                attack_advantage REAL,
                unlearning_status TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')

            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f" Table creation error: {e}")
            self.conn.rollback()
    
    def register_patient(self, patient, shard_id, slice_id,
                         feature_hash=None, model_hash=None):
        """
        Register a patient in the database with traceability information.
        
        Args:
            patient: Patient record object with patient_id, diagnosis, sequence
            shard_id (int): Shard ID
            slice_id (int): Slice ID
            feature_hash (str): Optional feature hash for tracing
            model_hash (str): Optional model hash for tracing
        """
        self._ensure_connection()
        
        # Generate hashes if not provided
        if feature_hash is None:
            feature_hash = hashlib.md5(
                str(getattr(patient, 'sequence', ''))[:100].encode()
            ).hexdigest()[:16]
        
        if model_hash is None:
            model_hash = hashlib.md5(
                f"{shard_id}_{slice_id}".encode()
            ).hexdigest()[:16]
        
        try:
            self.cursor.execute(
                '''
                INSERT OR REPLACE INTO patients
                (
                    patient_id,
                    diagnosis,
                    shard_id,
                    slice_id,
                    feature_hash,
                    model_parameters_hash
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    patient.patient_id,
                    patient.diagnosis,
                    shard_id,
                    slice_id,
                    feature_hash,
                    model_hash
                )
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f" Error registering patient {patient.patient_id}: {e}")
            self.conn.rollback()
            # Try to recover by reconnecting
            self._connect()
            self.register_patient(patient, shard_id, slice_id, feature_hash, model_hash)
    
    def register_shard(self, shard_id, num_samples, slice_count, model_path, accuracy):
        """
        Register shard information in the database.
        
        Args:
            shard_id (int): Shard ID
            num_samples (int): Number of samples in shard
            slice_count (int): Number of slices in shard
            model_path (str): Path to saved model
            accuracy (float): Shard model accuracy
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute(
                '''INSERT OR REPLACE INTO shards 
                   (shard_id, num_samples, slice_count, model_path, accuracy)
                   VALUES (?, ?, ?, ?, ?)''',
                (shard_id, num_samples, slice_count, model_path, accuracy)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f" Error registering shard {shard_id}: {e}")
            self.conn.rollback()
    
    def log_deletion(self, patient_id, shard_id, slice_id, time_taken, accuracy_drop):
        """
        Log a patient deletion/unlearning event.
        
        Args:
            patient_id (str): Patient ID
            shard_id (int): Shard ID
            slice_id (int): Slice ID
            time_taken (float): Time taken for unlearning in seconds
            accuracy_drop (float): Accuracy drop after unlearning
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute(
                '''INSERT INTO deletion_log
                   (patient_id, original_shard_id, original_slice_id, time_taken, accuracy_drop)
                   VALUES (?, ?, ?, ?, ?)''',
                (patient_id, shard_id, slice_id, time_taken, accuracy_drop)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f" Error logging deletion for {patient_id}: {e}")
            self.conn.rollback()
    
    def log_performance(self, fold, accuracy_before, accuracy_after, attack_advantage, status):
        """
        Log performance metrics for a fold.
        
        Args:
            fold (int): Fold number
            accuracy_before (float): Accuracy before unlearning
            accuracy_after (float): Accuracy after unlearning
            attack_advantage (float): Privacy attack advantage
            status (str): Unlearning status (SUCCESSFUL/PARTIAL/FAILED)
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute(
                '''INSERT INTO performance 
                   (fold, accuracy_before, accuracy_after, attack_advantage, unlearning_status)
                   VALUES (?, ?, ?, ?, ?)''',
                (fold, accuracy_before, accuracy_after, attack_advantage, status)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f" Error logging performance: {e}")
            self.conn.rollback()
    
    def get_patient_trace(self, patient_id):
        """
        Get traceability information for a patient.
        
        Args:
            patient_id (str): Patient ID
            
        Returns:
            tuple: (patient_id, diagnosis, shard_id, slice_id, feature_hash, 
                   model_parameters_hash, created_at)
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute(
                '''SELECT patient_id, diagnosis, shard_id, slice_id, 
                          feature_hash, model_parameters_hash, created_at
                   FROM patients WHERE patient_id = ?''',
                (patient_id,)
            )
            result = self.cursor.fetchone()
            return result if result else None
        except sqlite3.Error:
            return None
    
    def get_deletion_stats(self):
        """
        Get deletion statistics.
        
        Returns:
            dict: Statistics including total_deleted and avg_accuracy_drop
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute('SELECT COUNT(*) FROM deletion_log')
            deleted = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT AVG(accuracy_drop) FROM deletion_log')
            avg_drop = self.cursor.fetchone()[0]
            
            return {
                'total_deleted': deleted, 
                'avg_accuracy_drop': avg_drop if avg_drop else 0
            }
        except sqlite3.Error:
            return {'total_deleted': 0, 'avg_accuracy_drop': 0}
    
    def get_stats(self):
        """
        Get database statistics.
        
        Returns:
            dict: Statistics including total, deleted, and active patients
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute('SELECT COUNT(*) FROM patients')
            total = self.cursor.fetchone()[0]
            
            self.cursor.execute('SELECT COUNT(*) FROM deletion_log')
            deleted = self.cursor.fetchone()[0]
            
            return {
                'total': total, 
                'deleted': deleted, 
                'active': total - deleted
            }
        except sqlite3.Error:
            return {'total': 0, 'deleted': 0, 'active': 0}
    
    def get_performance_summary(self):
        """
        Get performance summary across all folds.
        
        Returns:
            dict: Performance summary
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute('''
                SELECT 
                    COUNT(*) as num_folds,
                    AVG(accuracy_before) as avg_acc_before,
                    AVG(accuracy_after) as avg_acc_after,
                    AVG(attack_advantage) as avg_attack_advantage,
                    COUNT(CASE WHEN unlearning_status = 'SUCCESSFUL' THEN 1 END) as successful_count
                FROM performance
            ''')
            result = self.cursor.fetchone()
            
            if result:
                return {
                    'num_folds': result[0],
                    'avg_accuracy_before': result[1] if result[1] else 0,
                    'avg_accuracy_after': result[2] if result[2] else 0,
                    'avg_attack_advantage': result[3] if result[3] else 0,
                    'successful_count': result[4] if result[4] else 0
                }
            return {}
        except sqlite3.Error:
            return {}
    
    def clear_all(self):
        """
        Clear all data from the database (for fresh start).
        """
        self._ensure_connection()
        
        try:
            self.cursor.execute("DELETE FROM patients")
            self.cursor.execute("DELETE FROM shards")
            self.cursor.execute("DELETE FROM deletion_log")
            self.cursor.execute("DELETE FROM feature_contributions")
            self.cursor.execute("DELETE FROM performance")
            self.conn.commit()
            print(" Database cleared")
        except sqlite3.Error as e:
            print(f" Error clearing database: {e}")
            self.conn.rollback()
    
    def close(self):
        """
        Close database connection safely.
        """
        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()
                self.is_open = False
                self.conn = None
                self.cursor = None
        except sqlite3.Error as e:
            print(f" Error closing database: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self._ensure_connection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection is closed."""
        self.close()
    
    def __del__(self):
        """Destructor - ensure connection is closed."""
        self.close()


# =================================================================================
# UTILITY FUNCTIONS
# =================================================================================

def reset_database(db_path="bio_forget_complete.db"):
    """
    Reset the database by removing the file.
    
    Args:
        db_path (str): Path to database file
        
    Returns:
        bool: True if reset successful
    """
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f" Database reset: {db_path}")
            return True
        except Exception as e:
            print(f" Could not reset database: {e}")
            return False
    return True


def get_database_info(db_path="bio_forget_complete.db"):
    """
    Get information about the database file.
    
    Args:
        db_path (str): Path to database file
        
    Returns:
        dict: Database information
    """
    info = {
        'exists': os.path.exists(db_path),
        'path': db_path
    }
    
    if info['exists']:
        info['size_kb'] = os.path.getsize(db_path) / 1024
        info['modified'] = datetime.fromtimestamp(os.path.getmtime(db_path))
    
    return info


# =================================================================================
# MAIN ENTRY POINT (for testing)
# =================================================================================

if __name__ == "__main__":
    # Test database functionality
    print("Testing EnhancedDatabase...")
    
    # Create database
    db = EnhancedDatabase("test.db", use_memory=True)
    
    # Create a mock patient
    class MockPatient:
        def __init__(self, pid, diagnosis, sequence=""):
            self.patient_id = pid
            self.diagnosis = diagnosis
            self.sequence = sequence
    
    patient = MockPatient("TEST_001", 1, "ATCGATCG")
    
    # Register patient
    db.register_patient(patient, shard_id=0, slice_id=1)
    
    # Log deletion
    db.log_deletion("TEST_001", shard_id=0, slice_id=1, time_taken=1.5, accuracy_drop=0.5)
    
    # Get stats
    stats = db.get_stats()
    print(f"Stats: {stats}")
    
    # Get trace
    trace = db.get_patient_trace("TEST_001")
    print(f"Trace: {trace}")
    
    # Close
    db.close()
    
    print(" Database test completed!")