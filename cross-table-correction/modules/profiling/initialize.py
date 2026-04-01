"""
Initializer function for the profiling module.
"""

import logging
import os
import time

from core.lake import Lake
from modules.profiling.build_profiles import build_column_profiles
from modules.profiling.io import get_profile_summary, save_profiles


def initialize_profiles(lake: Lake, output_dir: str):
    """
    Build clean-only column profiles for the lake and persist them.

    Args:
        lake: Lake object containing tables with clean/dirty data
        output_dir: Directory to save the profiles
        head_k: Number of top values to keep per column

    Returns:
        Tuple of (profiles_dict, profiles_file_path)
    """
    logging.info("Initializing column profiling module...")

    t0 = time.time()

    # Build profiles for all columns
    profiles = build_column_profiles(lake)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save profiles to disk
    profiles_path = os.path.join(output_dir, "profiles.pkl")
    save_profiles(profiles, profiles_path)

    # Log summary statistics
    summary = get_profile_summary(profiles)

    elapsed_time = time.time() - t0

    logging.info(f"Profiling initialization completed in {elapsed_time:.2f}s")
    logging.info(f"Profile summary: {summary}")
    logging.info(f"Profiles saved to: {profiles_path}")

    return profiles, profiles_path


def load_or_build_profiles(
    lake: Lake, output_dir: str, head_k: int = 100, force_rebuild: bool = False
):
    """
    Load existing profiles or build new ones if they don't exist.

    Args:
        lake: Lake object containing tables
        output_dir: Directory containing/to save profiles
        head_k: Number of top values to keep per column (used if building)
        force_rebuild: Whether to force rebuilding even if profiles exist

    Returns:
        Tuple of (profiles_dict, profiles_file_path)
    """
    from modules.profiling.io import load_profiles

    profiles_path = os.path.join(output_dir, "profiles.pkl")

    # Check if profiles already exist and we don't want to force rebuild
    if os.path.exists(profiles_path) and not force_rebuild:
        logging.info(f"Loading existing profiles from {profiles_path}")
        try:
            profiles = load_profiles(profiles_path)

            # Validate that profiles are compatible with current lake
            if _validate_profiles_compatibility(profiles, lake):
                summary = get_profile_summary(profiles)
                logging.info(f"Loaded profiles summary: {summary}")
                return profiles, profiles_path
            else:
                logging.warning(
                    "Existing profiles are incompatible with current lake, rebuilding..."
                )
        except Exception as e:
            logging.warning(f"Failed to load existing profiles: {e}, rebuilding...")

    # Build new profiles
    logging.info("Building new column profiles...")
    return initialize_profiles(lake, output_dir)


def _validate_profiles_compatibility(profiles, lake) -> bool:
    """
    Check if loaded profiles are compatible with the current lake structure.

    Args:
        profiles: Dictionary of loaded profiles
        lake: Current lake object

    Returns:
        True if compatible, False otherwise
    """
    # Get expected table/column combinations from lake
    expected_keys = set()
    for table_id, table in lake.tables.items():
        for column_idx in table.columns.keys():
            expected_keys.add((table_id, column_idx))

    # Get actual keys from profiles
    actual_keys = set(profiles.keys())

    # Check if they match (allowing for profiles to have extra columns)
    missing_keys = expected_keys - actual_keys

    if missing_keys:
        logging.warning(
            f"Profiles missing for {len(missing_keys)} columns: {list(missing_keys)[:5]}..."
        )
        return False

    # Check if we have a reasonable number of matching columns
    matching_keys = expected_keys & actual_keys
    match_ratio = len(matching_keys) / len(expected_keys) if expected_keys else 0

    if match_ratio < 0.8:  # At least 80% of columns should match
        logging.warning(
            f"Only {match_ratio:.1%} of columns match between profiles and lake"
        )
        return False

    logging.info(
        f"Profile compatibility check passed: {len(matching_keys)}/{len(expected_keys)} columns match"
    )
    return True


def update_profiles_after_sampling(profiles, lake, output_dir: str):
    """
    Update profiles after sampling to incorporate labeled data as additional clean examples.

    Args:
        profiles: Current profiles dictionary
        lake: Lake with updated sample information
        output_dir: Directory to save updated profiles

    Returns:
        Updated profiles dictionary
    """
    from modules.profiling.build_profiles import update_profiles_with_samples
    from modules.profiling.io import save_profiles

    logging.info("Updating profiles with labeled sample data...")

    t0 = time.time()

    # Update profiles with samples
    updated_profiles = update_profiles_with_samples(profiles, lake)

    # Save updated profiles
    profiles_path = os.path.join(output_dir, "profiles_with_samples.pkl")
    save_profiles(updated_profiles, profiles_path)

    elapsed_time = time.time() - t0

    logging.info(f"Profile update completed in {elapsed_time:.2f}s")
    logging.info(f"Updated profiles saved to: {profiles_path}")

    return updated_profiles
