"""
Save/load helpers for ColumnProfile dictionaries.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

from modules.profiling.column_profile import ColumnProfile


def save_profiles(
    profiles: Dict[Tuple[str, int], ColumnProfile], file_path: str
) -> None:
    """
    Save column profiles to disk.

    Args:
        profiles: Dictionary mapping (table_id, column_idx) to ColumnProfile
        file_path: Path to save the profiles
    """
    logging.info(f"Saving {len(profiles)} column profiles to {file_path}")

    # Convert to serializable format
    serializable_profiles = {}
    for key, profile in profiles.items():
        # Convert tuple key to string for JSON compatibility
        str_key = f"{key[0]}_{key[1]}"
        serializable_profiles[str_key] = profile.to_dict()

    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Save using pickle for now (can be changed to JSON if needed)
    with open(file_path, "wb") as f:
        pickle.dump(serializable_profiles, f)

    logging.info(f"Successfully saved profiles to {file_path}")


def load_profiles(file_path: str) -> Dict[Tuple[str, int], ColumnProfile]:
    """
    Load column profiles from disk.

    Args:
        file_path: Path to load the profiles from

    Returns:
        Dictionary mapping (table_id, column_idx) to ColumnProfile
    """
    logging.info(f"Loading column profiles from {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Profile file not found: {file_path}")

    with open(file_path, "rb") as f:
        serializable_profiles = pickle.load(f)

    # Convert back to proper format
    profiles = {}
    for str_key, profile_dict in serializable_profiles.items():
        # Parse string key back to tuple
        parts = str_key.split("_", 1)  # Split on first underscore only
        if len(parts) != 2:
            logging.warning(f"Invalid profile key format: {str_key}")
            continue

        table_id = parts[0]
        try:
            column_idx = int(parts[1])
        except ValueError:
            logging.warning(f"Invalid column index in key: {str_key}")
            continue

        key = (table_id, column_idx)
        profiles[key] = ColumnProfile.from_dict(profile_dict)

    logging.info(f"Successfully loaded {len(profiles)} column profiles")
    return profiles


def save_profiles_json(
    profiles: Dict[Tuple[str, int], ColumnProfile], file_path: str
) -> None:
    """
    Save column profiles to disk in JSON format (alternative to pickle).

    Args:
        profiles: Dictionary mapping (table_id, column_idx) to ColumnProfile
        file_path: Path to save the profiles (should end with .json)
    """
    import json

    logging.info(f"Saving {len(profiles)} column profiles to JSON: {file_path}")

    # Convert to serializable format
    serializable_profiles = {}
    for key, profile in profiles.items():
        str_key = f"{key[0]}_{key[1]}"
        serializable_profiles[str_key] = profile.to_dict()

    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_profiles, f, indent=2, ensure_ascii=False)

    logging.info(f"Successfully saved profiles to JSON: {file_path}")


def load_profiles_json(file_path: str) -> Dict[Tuple[str, int], ColumnProfile]:
    """
    Load column profiles from JSON file.

    Args:
        file_path: Path to load the profiles from

    Returns:
        Dictionary mapping (table_id, column_idx) to ColumnProfile
    """
    import json

    logging.info(f"Loading column profiles from JSON: {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Profile file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        serializable_profiles = json.load(f)

    # Convert back to proper format
    profiles = {}
    for str_key, profile_dict in serializable_profiles.items():
        parts = str_key.split("_", 1)
        if len(parts) != 2:
            logging.warning(f"Invalid profile key format: {str_key}")
            continue

        table_id = parts[0]
        try:
            column_idx = int(parts[1])
        except ValueError:
            logging.warning(f"Invalid column index in key: {str_key}")
            continue

        key = (table_id, column_idx)
        profiles[key] = ColumnProfile.from_dict(profile_dict)

    logging.info(f"Successfully loaded {len(profiles)} column profiles from JSON")
    return profiles


def get_profile_summary(profiles: Dict[Tuple[str, int], ColumnProfile]) -> Dict:
    """
    Get summary statistics about the profiles.

    Args:
        profiles: Dictionary of profiles

    Returns:
        Dictionary with summary statistics
    """
    if not profiles:
        return {"total_profiles": 0}

    total_profiles = len(profiles)
    total_patterns = sum(len(p.mask_histogram) for p in profiles.values())

    # Tables and columns
    tables = set()
    columns_per_table = {}

    for (table_id, column_idx), profile in profiles.items():
        tables.add(table_id)
        if table_id not in columns_per_table:
            columns_per_table[table_id] = 0
        columns_per_table[table_id] += 1

    avg_columns_per_table = (
        sum(columns_per_table.values()) / len(columns_per_table)
        if columns_per_table
        else 0
    )

    return {
        "total_profiles": total_profiles,
        "total_tables": len(tables),
        "avg_columns_per_table": round(avg_columns_per_table, 2),
        "total_patterns": total_patterns,
        "avg_patterns_per_column": round(total_patterns / total_profiles, 2),
    }
