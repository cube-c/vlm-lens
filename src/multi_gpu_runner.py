"""multi_gpu_runner.py

Multi-GPU orchestration wrapper for VLM-Lens.
Handles dataset splitting, process spawning, and database merging.
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_gpu_ids(gpu_str: str) -> List[int]:
    """Parse comma-separated GPU ID string into list of integers.

    Args:
        gpu_str: String like "0,1,2,3"

    Returns:
        List of GPU IDs

    Raises:
        ValueError: If GPU IDs are invalid or unavailable
    """
    try:
        gpu_ids = [int(x.strip()) for x in gpu_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid GPU ID format: {gpu_str}. Expected comma-separated integers.")

    # Validate GPU availability
    available_gpus = torch.cuda.device_count()
    if not available_gpus:
        raise ValueError("No GPUs available on this system")

    for gpu_id in gpu_ids:
        if gpu_id < 0 or gpu_id >= available_gpus:
            raise ValueError(
                f"GPU {gpu_id} not available. System has {available_gpus} GPUs (0-{available_gpus-1})"
            )

    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"Duplicate GPU IDs found in {gpu_str}")

    return gpu_ids


def calculate_dataset_splits(total_size: int, num_gpus: int) -> List[Tuple[int, int]]:
    """Calculate even dataset splits for each GPU.

    Args:
        total_size: Total number of samples in dataset
        num_gpus: Number of GPUs to split across

    Returns:
        List of (start_idx, end_idx) tuples for each GPU
    """
    base_size = total_size // num_gpus
    remainder = total_size % num_gpus

    splits = []
    current_idx = 0

    for i in range(num_gpus):
        # Distribute remainder samples to first few GPUs
        size = base_size + (1 if i < remainder else 0)
        start_idx = current_idx
        end_idx = current_idx + size
        splits.append((start_idx, end_idx))
        current_idx = end_idx

    return splits


def get_dataset_size(config_args: List[str]) -> int:
    """Determine total dataset size by temporarily loading config.

    This is necessary to calculate splits before spawning processes.

    Args:
        config_args: Original CLI arguments

    Returns:
        Total dataset size
    """
    # Import here to avoid circular dependencies
    from src.models.config import Config

    # Create a temporary config to get dataset size
    # We need to temporarily remove multi-GPU args to avoid conflicts
    temp_args = [arg for arg in config_args if not arg.startswith('--gpus')]

    # Temporarily override sys.argv
    original_argv = sys.argv.copy()
    sys.argv = ['src.main'] + temp_args

    try:
        config = Config()

        # Get size based on config setup
        if config.dataset:
            total_size = len(config.dataset)
        elif config.has_images():
            total_size = len(config.image_paths)
        else:
            total_size = 1  # Text-only prompt

        # Apply sample limit if set
        if hasattr(config, 'sample') and config.sample is not None:
            total_size = min(config.sample, total_size)

        return total_size
    finally:
        sys.argv = original_argv


def run_worker(
    gpu_id: int,
    start_idx: int,
    end_idx: int,
    output_db: str,
    original_args: List[str]
) -> None:
    """Worker process for a single GPU.

    Args:
        gpu_id: GPU ID to use
        start_idx: Starting dataset index
        end_idx: Ending dataset index
        output_db: Output database path for this GPU
        original_args: Original CLI arguments
    """
    # Set CUDA_VISIBLE_DEVICES to restrict this process to its GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import here to ensure it uses the right CUDA device
    from src.models.config import Config
    from src.main import get_model

    # Build modified arguments for this worker
    worker_args = original_args.copy()

    # Remove --gpus argument
    worker_args = [
        arg for i, arg in enumerate(worker_args)
        if not (arg == '--gpus' or (i > 0 and worker_args[i-1] == '--gpus'))
    ]

    # Add worker-specific arguments
    worker_args.extend([
        '--device', 'cuda',  # Since CUDA_VISIBLE_DEVICES is set, cuda:0 is our GPU
        '--start-idx', str(start_idx),
        '--end-idx', str(end_idx),
        '--output-db', output_db,
        '--gpu-id', str(gpu_id)
    ])

    # Override sys.argv for config parsing
    sys.argv = ['src.main'] + worker_args

    # Run the model
    logging.info(f"GPU {gpu_id}: Processing samples {start_idx}-{end_idx} -> {output_db}")

    try:
        config = Config()
        model = get_model(config.architecture, config)
        model.run()
        logging.info(f"GPU {gpu_id}: Completed successfully")
    except Exception as e:
        logging.error(f"GPU {gpu_id}: Failed with error: {e}")
        raise


def merge_databases(db_paths: List[str], final_output: str) -> None:
    """Merge multiple SQLite databases into one.

    Args:
        db_paths: List of database paths to merge
        final_output: Final output database path
    """
    logging.info(f"Merging {len(db_paths)} databases into {final_output}")

    # Create final database
    final_conn = sqlite3.connect(final_output)
    final_cursor = final_conn.cursor()

    # Create table structure (copy from first DB)
    if os.path.exists(db_paths[0]):
        temp_conn = sqlite3.connect(db_paths[0])
        temp_cursor = temp_conn.cursor()

        # Get CREATE TABLE statement
        temp_cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='tensors'"
        )
        result = temp_cursor.fetchone()
        if result:
            create_statement = result[0]
            final_cursor.execute(create_statement)
        temp_conn.close()

    # Merge all databases
    for db_path in db_paths:
        if not os.path.exists(db_path):
            logging.warning(f"Database {db_path} not found, skipping")
            continue

        logging.info(f"Merging {db_path}...")

        # Attach database
        final_cursor.execute(f"ATTACH DATABASE '{db_path}' AS source_db")

        # Copy all rows (excluding id to let SQLite auto-increment)
        final_cursor.execute("""
            INSERT INTO tensors (name, architecture, timestamp, image_path, image_id,
                                prompt, label, layer, pooling_method, tensor_dim, tensor)
            SELECT name, architecture, timestamp, image_path, image_id,
                   prompt, label, layer, pooling_method, tensor_dim, tensor
            FROM source_db.tensors
        """)

        # Detach database
        final_cursor.execute("DETACH DATABASE source_db")

        final_conn.commit()

    # Get final row count
    final_cursor.execute("SELECT COUNT(*) FROM tensors")
    total_rows = final_cursor.fetchone()[0]

    final_conn.close()

    logging.info(f"Merge complete: {total_rows} total rows in {final_output}")


def cleanup_temp_databases(db_paths: List[str], keep_temp: bool = False) -> None:
    """Clean up temporary per-GPU database files.

    Args:
        db_paths: List of database paths to remove
        keep_temp: If True, keep temporary databases
    """
    if keep_temp:
        logging.info("Keeping temporary database files")
        return

    logging.info("Cleaning up temporary database files...")
    for db_path in db_paths:
        if os.path.exists(db_path):
            os.remove(db_path)
            logging.info(f"Removed {db_path}")


def main():
    """Main entry point for multi-GPU runner."""
    # Parse only the --gpus argument and pass rest through
    parser = argparse.ArgumentParser(description='Multi-GPU runner for VLM-Lens')
    parser.add_argument('--gpus', type=str, required=True,
                       help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--keep-temp-dbs', action='store_true',
                       help='Keep temporary per-GPU database files after merging')

    # Parse known args, keep unknown for passing through
    args, unknown_args = parser.parse_known_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpus)
    num_gpus = len(gpu_ids)

    logging.info(f"Multi-GPU mode: Using {num_gpus} GPUs: {gpu_ids}")

    # Get dataset size
    total_size = get_dataset_size(unknown_args)
    logging.info(f"Total dataset size: {total_size} samples")

    if total_size < num_gpus:
        logging.warning(
            f"Dataset size ({total_size}) is smaller than number of GPUs ({num_gpus}). "
            f"Some GPUs will be idle."
        )

    # Calculate dataset splits
    splits = calculate_dataset_splits(total_size, num_gpus)

    # Determine output database path (from args or default)
    output_db = 'embeddings.db'
    for i, arg in enumerate(unknown_args):
        if arg in ['--output-db', '-o'] and i + 1 < len(unknown_args):
            output_db = unknown_args[i + 1]
            break

    # Generate per-GPU database paths
    base_name, ext = os.path.splitext(output_db)
    gpu_db_paths = [f"{base_name}_gpu{gpu_id}{ext}" for gpu_id in gpu_ids]

    # Log execution plan
    logging.info("Execution plan:")
    for gpu_id, (start, end), db_path in zip(gpu_ids, splits, gpu_db_paths):
        logging.info(f"  GPU {gpu_id}: samples {start}-{end} ({end-start} samples) -> {db_path}")

    # Spawn worker processes
    logging.info("Spawning worker processes...")

    # Use spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    processes = []
    for gpu_id, (start_idx, end_idx), db_path in zip(gpu_ids, splits, gpu_db_paths):
        if start_idx >= end_idx:
            logging.info(f"GPU {gpu_id}: No samples assigned, skipping")
            continue

        p = mp.Process(
            target=run_worker,
            args=(gpu_id, start_idx, end_idx, db_path, unknown_args)
        )
        p.start()
        processes.append((p, gpu_id))

    # Wait for all processes to complete
    logging.info("Waiting for all workers to complete...")

    failed_gpus = []
    for p, gpu_id in processes:
        p.join()
        if p.exitcode != 0:
            logging.error(f"GPU {gpu_id} process failed with exit code {p.exitcode}")
            failed_gpus.append(gpu_id)

    if failed_gpus:
        logging.error(f"Failed GPUs: {failed_gpus}")
        logging.error("Some workers failed. Check logs above for details.")
        sys.exit(1)

    logging.info("All workers completed successfully")

    # Merge databases
    merge_databases(gpu_db_paths, output_db)

    # Cleanup temporary databases
    cleanup_temp_databases(gpu_db_paths, keep_temp=args.keep_temp_dbs)

    logging.info(f"Multi-GPU processing complete! Final database: {output_db}")


if __name__ == '__main__':
    main()
