#!/usr/bin/env python3
"""
Majority voting evaluation for Uniclean on the joined soccer table.

Algorithm (per column):
  1. Determine provenance: column_provenance.json → split_a or split_b,
     then provenance_map.csv → original isolated row ID for each joined row.
  2. Identify errors: cells where dirty ≠ clean (true errors in the joined table).
  3. Collect votes: for each unique original cell (split, split_row_id, col),
     gather all Uniclean-repaired values from every joined row that references it
     (only where repaired ≠ dirty, i.e. Uniclean changed something).
  4. Majority vote: most-frequent repaired value; ties broken lexicographically.
  5. Evaluate: voted value vs clean value → TP or FP.

Usage:
    python evaluate_joined_soccer_majority_voting_uniclean.py \\
        --soccer-dir datasets_and_rules/joined_soccer/soccer \\
        --table-name soccer_joined_fixed_prov \\
        --save-csv datasets_and_rules/joined_soccer/soccer/result/soccer_joined_fixed_prov/majority_voting_uniclean.csv
"""
from pathlib import Path
f