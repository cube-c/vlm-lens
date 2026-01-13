"""Convert RoboSpatial JSON data to HuggingFace datasets arrow format."""

import argparse
from pathlib import Path

from datasets import Dataset, Image, load_dataset


def remap_image_path(example: dict, old_prefix: str, new_prefix: str) -> dict:
    """Remap image path from old prefix to new prefix.

    Args:
        example: A dataset example with 'image_path' field
        old_prefix: The old path prefix to replace
        new_prefix: The new path prefix

    Returns:
        Modified example with updated image_path
    """
    if "image_path" in example and example["image_path"].startswith(old_prefix):
        # Replace old prefix with new prefix
        relative_path = example["image_path"][len(old_prefix):]
        example["image_path"] = new_prefix + relative_path
    return example


def main(
    json_file: str,
    output_dir: str,
    use_image_feature: bool = False,
    image_base_path: str = None
) -> None:
    """Convert RoboSpatial JSON to arrow format.

    Args:
        json_file: Path to the JSON file
        output_dir: Directory to save the arrow dataset
        use_image_feature: Whether to cast image_path to Image feature type
        image_base_path: Base path where images are actually located (for path remapping)
    """
    print(f"Loading dataset from {json_file}...")

    # Load the JSON file as a dataset
    ds = load_dataset("json", data_files=json_file, split="train")

    print(f"Loaded {len(ds)} samples")
    print(f"Columns: {ds.column_names}")
    print(f"First sample (before remapping): {ds[0]}")

    # Remap image paths if needed
    if image_base_path:
        print(f"Remapping image paths to base: {image_base_path}")
        old_prefix = "/media/luke/storage/datasets/robospatial/images/scannet"
        new_prefix = image_base_path.rstrip("/") + "/images/"

        ds = ds.map(
            lambda example: remap_image_path(example, old_prefix, new_prefix),
            desc="Remapping image paths"
        )
        print(f"First sample (after remapping): {ds[0]}")

    # Rename columns for consistency
    print("Renaming columns: question → prompt, answer → label")
    if "question" in ds.column_names:
        ds = ds.rename_column("question", "prompt")
    if "answer" in ds.column_names:
        ds = ds.rename_column("answer", "label")
    if "image_path" in ds.column_names:
        ds = ds.rename_column("image_path", "image")

    # Add auto-incrementing ID column
    print("Adding auto-incrementing ID column...")
    ds = ds.map(
        lambda example, idx: {"id": idx},
        with_indices=True,
        desc="Adding ID column"
    )

    # Convert label from Yes/No to boolean
    if "label" in ds.column_names:
        print("Converting label from Yes/No to boolean...")
        ds = ds.map(
            lambda example: {"label": example["label"].lower() == "yes"},
            desc="Converting labels to boolean"
        )

    # Optional: Cast image_path column to Image feature
    # This keeps paths but decodes images on access
    if use_image_feature:
        # print("Casting image_path column to Image feature...")
        ds = ds.cast_column("image", Image())

    # Save to disk in arrow format
    print(f"Saving dataset to {output_dir}...")
    ds.save_to_disk(output_dir)

    print(f"✓ Dataset saved successfully!")
    print(f"  - Location: {output_dir}")
    print(f"  - Samples: {len(ds)}")
    print(f"  - Features: {ds.features}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RoboSpatial JSON to HuggingFace arrow format"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="robospatial/robospatial_compatibility.json",
        help="Path to the RoboSpatial JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/robospatial/compatibility",
        help="Directory to save the arrow dataset"
    )
    parser.add_argument(
        "--use-image-feature",
        action="store_true",
        help="Cast image_path to Image feature (decodes images on access)"
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        default="/data/shared/Qwen/RoboSpatial_data",
        help="Base path where images are actually located (for path remapping)"
    )

    args = parser.parse_args()

    main(
        json_file=args.json_file,
        output_dir=args.output_dir,
        use_image_feature=args.use_image_feature,
        image_base_path=args.image_base_path
    )
