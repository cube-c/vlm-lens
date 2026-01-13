"""Unit tests for multi-GPU functionality."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.multi_gpu_runner import (
    parse_gpu_ids,
    calculate_dataset_splits,
    merge_databases,
)


class TestParseGPUIDs:
    """Test GPU ID parsing functionality."""

    def test_parse_single_gpu(self):
        """Test parsing single GPU ID."""
        # Mock torch.cuda.device_count
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA available for testing")

        result = parse_gpu_ids("0")
        assert result == [0]

    def test_parse_multiple_gpus(self):
        """Test parsing multiple GPU IDs."""
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip("Insufficient GPUs for testing")

        result = parse_gpu_ids("0,1,2,3")
        assert result == [0, 1, 2, 3]

    def test_parse_with_spaces(self):
        """Test parsing GPU IDs with spaces."""
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            pytest.skip("Insufficient GPUs for testing")

        result = parse_gpu_ids("0, 1, 2")
        assert result == [0, 1, 2]

    def test_parse_non_sequential(self):
        """Test parsing non-sequential GPU IDs."""
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            pytest.skip("Insufficient GPUs for testing")

        result = parse_gpu_ids("3,1,0,2")
        assert result == [3, 1, 0, 2]

    def test_parse_invalid_format(self):
        """Test parsing with invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GPU ID format"):
            parse_gpu_ids("0,1,abc")

    def test_parse_duplicate_gpus(self):
        """Test parsing with duplicate GPU IDs raises ValueError."""
        with pytest.raises(ValueError, match="Duplicate GPU IDs"):
            parse_gpu_ids("0,1,1,2")

    def test_parse_negative_gpu_id(self):
        """Test parsing with negative GPU ID raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            parse_gpu_ids("-1,0,1")

    def test_parse_gpu_out_of_range(self):
        """Test parsing with GPU ID out of range raises ValueError."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No CUDA available for testing")

        gpu_count = torch.cuda.device_count()
        with pytest.raises(ValueError, match="not available"):
            parse_gpu_ids(f"0,{gpu_count}")


class TestCalculateDatasetSplits:
    """Test dataset splitting functionality."""

    def test_even_split(self):
        """Test even dataset split."""
        splits = calculate_dataset_splits(100, 4)
        assert splits == [(0, 25), (25, 50), (50, 75), (75, 100)]

    def test_uneven_split(self):
        """Test uneven dataset split - remainder distributed to first GPU."""
        splits = calculate_dataset_splits(10, 3)
        assert splits == [(0, 4), (4, 7), (7, 10)]
        # First GPU gets 4 (base 3 + remainder 1), second and third get 3

    def test_uneven_split_with_larger_remainder(self):
        """Test uneven split with larger remainder."""
        splits = calculate_dataset_splits(103, 4)
        assert splits == [(0, 26), (26, 52), (52, 78), (78, 103)]
        # First 3 GPUs get 26 samples, last GPU gets 25

    def test_more_gpus_than_samples(self):
        """Test when there are more GPUs than samples."""
        splits = calculate_dataset_splits(2, 4)
        assert splits == [(0, 1), (1, 2), (2, 2), (2, 2)]
        # Last 2 GPUs get empty ranges

    def test_single_gpu(self):
        """Test split with single GPU."""
        splits = calculate_dataset_splits(100, 1)
        assert splits == [(0, 100)]

    def test_single_sample(self):
        """Test split with single sample."""
        splits = calculate_dataset_splits(1, 1)
        assert splits == [(0, 1)]

    def test_zero_samples(self):
        """Test split with zero samples."""
        splits = calculate_dataset_splits(0, 4)
        assert splits == [(0, 0), (0, 0), (0, 0), (0, 0)]

    def test_exact_division(self):
        """Test when dataset size is exactly divisible by GPU count."""
        splits = calculate_dataset_splits(200, 8)
        assert splits == [
            (0, 25), (25, 50), (50, 75), (75, 100),
            (100, 125), (125, 150), (150, 175), (175, 200)
        ]


class TestMergeDatabases:
    """Test database merging functionality."""

    def create_test_db(self, db_path: str, num_rows: int) -> None:
        """Helper to create a test database with sample data."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE tensors (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                architecture TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT NULL,
                image_id INTEGER NULL,
                prompt TEXT NOT NULL,
                label TEXT NULL,
                layer TEXT NOT NULL,
                pooling_method TEXT NULL,
                tensor_dim INTEGER NOT NULL,
                tensor BLOB NOT NULL
            )
        """)

        # Insert test rows
        for i in range(num_rows):
            cursor.execute("""
                INSERT INTO tensors (name, architecture, image_path, image_id, prompt, label, layer, pooling_method, tensor_dim, tensor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'test_model', 'test_arch', f'/path/to/image{i}.jpg', i,
                f'test prompt {i}', 'test_label', f'layer_{i}', 'mean',
                512, b'fake_tensor_data'
            ))

        conn.commit()
        conn.close()

    def test_merge_two_databases(self):
        """Test merging two databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1_path = os.path.join(tmpdir, "db1.db")
            db2_path = os.path.join(tmpdir, "db2.db")
            merged_path = os.path.join(tmpdir, "merged.db")

            # Create test databases
            self.create_test_db(db1_path, 5)
            self.create_test_db(db2_path, 3)

            # Merge databases
            total_rows = merge_databases([db1_path, db2_path], merged_path)

            # Verify
            assert total_rows == 8

            conn = sqlite3.connect(merged_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tensors")
            count = cursor.fetchone()[0]
            assert count == 8
            conn.close()

    def test_merge_multiple_databases(self):
        """Test merging multiple databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_paths = [os.path.join(tmpdir, f"db{i}.db") for i in range(4)]
            merged_path = os.path.join(tmpdir, "merged.db")

            # Create test databases with different row counts
            for i, db_path in enumerate(db_paths):
                self.create_test_db(db_path, (i + 1) * 2)  # 2, 4, 6, 8 rows

            # Merge databases
            total_rows = merge_databases(db_paths, merged_path)

            # Verify (2 + 4 + 6 + 8 = 20)
            assert total_rows == 20

    def test_merge_with_missing_source(self):
        """Test merging when one source database is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1_path = os.path.join(tmpdir, "db1.db")
            db2_path = os.path.join(tmpdir, "db2.db")  # This won't be created
            merged_path = os.path.join(tmpdir, "merged.db")

            # Create only first database
            self.create_test_db(db1_path, 5)

            # Merge databases (should skip missing db2)
            total_rows = merge_databases([db1_path, db2_path], merged_path)

            # Verify (only db1 rows)
            assert total_rows == 5

    def test_merge_empty_database(self):
        """Test merging an empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1_path = os.path.join(tmpdir, "db1.db")
            db2_path = os.path.join(tmpdir, "db2.db")
            merged_path = os.path.join(tmpdir, "merged.db")

            # Create databases
            self.create_test_db(db1_path, 5)
            self.create_test_db(db2_path, 0)  # Empty

            # Merge databases
            total_rows = merge_databases([db1_path, db2_path], merged_path)

            # Verify
            assert total_rows == 5

    def test_merge_preserves_data(self):
        """Test that merge preserves all data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1_path = os.path.join(tmpdir, "db1.db")
            merged_path = os.path.join(tmpdir, "merged.db")

            # Create test database
            self.create_test_db(db1_path, 3)

            # Merge single database
            merge_databases([db1_path], merged_path)

            # Verify data integrity
            conn = sqlite3.connect(merged_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, architecture, prompt, layer FROM tensors ORDER BY image_id")
            rows = cursor.fetchall()

            assert len(rows) == 3
            assert rows[0][2] == 'test prompt 0'
            assert rows[1][2] == 'test prompt 1'
            assert rows[2][2] == 'test prompt 2'
            conn.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
