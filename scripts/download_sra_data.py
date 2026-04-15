#!/usr/bin/env python3
"""
Download SRA data from NCBI Sequence Read Archive.

This script:
- Reads all CSV files in metadata/ to find sample_ids matching SRR*, ERR*, or DRR* pattern
- Uses each metadata filename stem as study_name (e.g., AAM+13.csv -> AAM+13)
- Downloads full gzip files from NCBI SRA URLs
- Verifies gzip integrity for every downloaded file
- Saves to data/fasta/<study_name>/<sample_id>.fasta.gz
- Skips existing files
"""

import csv
import gzip
import re
import signal
import sys
from pathlib import Path
from typing import Set

import requests

# Global variables for cleanup
temp_files: Set[Path] = set()
interrupted = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by cleaning up temp files."""
    global interrupted
    interrupted = True
    print("\n\nInterrupted! Cleaning up temporary files...")
    cleanup_temp_files()
    sys.exit(1)


def cleanup_temp_files():
    """Remove all temporary files."""
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            print(f"Warning: Could not remove {temp_file}: {e}")
    temp_files.clear()


def download_full_gzip(url: str, output_path: Path) -> bool:
    """Download a full gzip file from a URL."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return True
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


def verify_gzip_integrity(gzip_path: Path) -> bool:
    """Verify that a gzip file can be read to EOF."""
    try:
        with gzip.open(gzip_path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception as e:
        print(f"    Error verifying gzip integrity: {e}")
        return False


def process_sample(study_name: str, sample_id: str, output_dir: Path) -> bool:
    """
    Download and process a single SRA sample.

    Returns True if successful, False otherwise.
    """
    # Check if output file already exists
    output_file = output_dir / f"{sample_id}.fasta.gz"
    if output_file.exists():
        return True  # Already downloaded

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct download URL
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-reads-be/fasta?acc={sample_id}"

    # Download to a temporary file in the final output directory.
    temp_gzip = output_file.with_suffix(".fasta.gz.part")
    temp_files.add(temp_gzip)

    try:
        # Step 1: Download full gzip file
        if not download_full_gzip(url, temp_gzip):
            return False

        # Step 2: Verify gzip integrity
        if not verify_gzip_integrity(temp_gzip):
            return False

        # Step 3: Move to final output path
        temp_gzip.replace(output_file)

        return True

    except Exception as e:
        print(f"    Error processing {sample_id}: {e}")
        return False
    finally:
        # Clean up temporary files
        for temp_file in [temp_gzip]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    temp_files.discard(temp_file)
            except Exception:
                pass


def main():
    """Main function to process all samples from metadata CSV files."""
    global interrupted

    # Set up signal handler for cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    metadata_dir = repo_root / "metadata"
    fasta_base_dir = repo_root / "data" / "fasta"

    if not metadata_dir.exists():
        print(f"Error: {metadata_dir} not found")
        sys.exit(1)

    # Pattern to match SRR*, ERR*, or DRR* sample_ids
    pattern = re.compile(r"^(SRR|ERR|DRR)\d+$")

    # Statistics
    total_samples = 0
    skipped_pattern = 0
    skipped_existing = 0
    downloaded = 0
    failed = 0

    metadata_files = sorted(metadata_dir.glob("*.csv"))
    if not metadata_files:
        print(f"Error: no CSV files found in {metadata_dir}")
        sys.exit(1)

    print(f"Reading metadata files from {metadata_dir}...")
    try:
        for csv_path in metadata_files:
            if interrupted:
                break

            study_name = csv_path.stem

            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            print(f"Found {len(rows)} rows in {csv_path.name} (study {study_name})")

            for row in rows:
                if interrupted:
                    break

                sample_id = row.get("sample_id", "").strip()

                # Skip if doesn't match pattern
                if not pattern.match(sample_id):
                    skipped_pattern += 1
                    continue

                total_samples += 1

                # Check if already downloaded
                output_dir = fasta_base_dir / study_name
                output_file = output_dir / f"{sample_id}.fasta.gz"
                if output_file.exists():
                    skipped_existing += 1
                    continue

                # Process sample
                print(f"[{total_samples}] Processing {study_name}/{sample_id}...")
                if process_sample(study_name, sample_id, output_dir):
                    downloaded += 1
                    print("    \u2713 Downloaded and verified")
                else:
                    failed += 1
                    print("    \u2717 Failed")

        # Final cleanup
        cleanup_temp_files()

        # Print summary
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total samples matching pattern: {total_samples}")
        print(f"  Downloaded: {downloaded}")
        print(f"  Skipped (already exists): {skipped_existing}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (wrong pattern): {skipped_pattern}")
        print("=" * 60)

    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\nError: {e}")
        cleanup_temp_files()
        sys.exit(1)


if __name__ == "__main__":
    main()
