"""Simple verification script for multi-GPU functionality (without pytest dependency)."""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.multi_gpu_runner import parse_gpu_ids, calculate_dataset_splits


def test_parse_gpu_ids():
    """Test GPU ID parsing."""
    print("Testing parse_gpu_ids...")

    # Test with mock (will fail if no GPUs, but we can catch it)
    try:
        import torch
        if torch.cuda.is_available():
            result = parse_gpu_ids("0,1,2")
            print(f"  ✓ parse_gpu_ids('0,1,2') = {result}")
            assert result == [0, 1, 2], "Failed to parse GPU IDs correctly"
        else:
            print("  ⊘ Skipped (no CUDA available)")
    except ValueError as e:
        print(f"  ⊘ Skipped (insufficient GPUs): {e}")


def test_calculate_dataset_splits():
    """Test dataset splitting."""
    print("\nTesting calculate_dataset_splits...")

    # Test even split
    splits = calculate_dataset_splits(100, 4)
    expected = [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert splits == expected, f"Even split failed: got {splits}, expected {expected}"
    print(f"  ✓ Even split (100 samples, 4 GPUs): {splits}")

    # Test uneven split (10 // 3 = 3 base, 10 % 3 = 1 remainder)
    # First GPU gets base + 1 = 4, others get base = 3
    splits = calculate_dataset_splits(10, 3)
    expected = [(0, 4), (4, 7), (7, 10)]
    assert splits == expected, f"Uneven split failed: got {splits}, expected {expected}"
    print(f"  ✓ Uneven split (10 samples, 3 GPUs): {splits}")

    # Test more GPUs than samples
    splits = calculate_dataset_splits(2, 4)
    expected = [(0, 1), (1, 2), (2, 2), (2, 2)]
    assert splits == expected, f"More GPUs than samples failed: got {splits}, expected {expected}"
    print(f"  ✓ More GPUs than samples (2 samples, 4 GPUs): {splits}")

    # Test single GPU
    splits = calculate_dataset_splits(100, 1)
    expected = [(0, 100)]
    assert splits == expected, f"Single GPU failed: got {splits}, expected {expected}"
    print(f"  ✓ Single GPU (100 samples, 1 GPU): {splits}")


def test_invalid_gpu_ids():
    """Test invalid GPU ID handling."""
    print("\nTesting invalid GPU ID handling...")

    # Test invalid format
    try:
        parse_gpu_ids("0,1,abc")
        assert False, "Should have raised ValueError for invalid format"
    except ValueError as e:
        print(f"  ✓ Invalid format caught: {str(e)[:50]}...")

    # Test duplicates
    try:
        parse_gpu_ids("0,1,1,2")
        assert False, "Should have raised ValueError for duplicates"
    except ValueError as e:
        print(f"  ✓ Duplicate IDs caught: {str(e)[:50]}...")


if __name__ == '__main__':
    print("=" * 60)
    print("Multi-GPU Functionality Verification")
    print("=" * 60)

    try:
        test_calculate_dataset_splits()
        test_parse_gpu_ids()
        test_invalid_gpu_ids()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
