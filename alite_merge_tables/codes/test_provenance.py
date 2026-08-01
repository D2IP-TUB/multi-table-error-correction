"""
test_provenance.py

Runs FDAlgorithm on the minimum_example/integration_set and prints the FD
output, cell-level source map, and subsumption map in a readable way.

Run from the codes/ directory:
    python test_provenance.py

Set a breakpoint on any print() below to step through in the debugger.
"""

import glob
import os
import sys

# Make sure we can import alite_fd from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alite_fd import FDAlgorithm

CLUSTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "minimum_example", "integration_set")
CLUSTER_NAME = "integration_set"

filenames = sorted(glob.glob(os.path.join(CLUSTER_DIR, "*.csv")))
print("Input tables:")
for f in filenames:
    print(f"  {os.path.basename(f)}")
print()

# ── Run the algorithm ──────────────────────────────────────────────────────────
fd_table, cell_map, sub_map, stats, _ = FDAlgorithm(filenames, CLUSTER_NAME)

# ── 1. FD output ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FD OUTPUT  (one row per output tuple)")
print("="*70)
print(fd_table.to_string(index=True))

# ── 2. Subsumption map ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUBSUMPTION MAP  (every source row absorbed into each output row)")
print("  output_row_number | source_table | source_row")
print("="*70)
for out_row in sorted(sub_map["output_row_number"].unique()):
    subset = sub_map[sub_map["output_row_number"] == out_row]
    fd_vals = fd_table.iloc[out_row].to_dict()
    print(f"\n  Output row {out_row}: {fd_vals}")
    for _, r in subset.iterrows():
        print(f"    ← {r['source_table']}  row {r['source_row']}")

# ── 3. Cell-level source map ──────────────────────────────────────────────────
print("\n" + "="*70)
print("CELL SOURCE MAP  (which source row provided each non-null output cell)")
print("  output_row | column | value | source_table | source_row | error_type")
print("="*70)
for _, r in cell_map.iterrows():
    out_val = fd_table.iloc[r["row_number"]][r["column_name"]]
    print(f"  row {r['row_number']:>2}  col={r['column_name']:<12}  "
          f"val={str(out_val):<20}  "
          f"from {r['source_table']} row {r['source_row']}")

# ── 4. Highlight the interesting cases ───────────────────────────────────────
print("\n" + "="*70)
print("INTERESTING CASES")
print("="*70)

# Sofi Stadium — t5 row should be subsumed (its Team was empty)
sofi_mask = fd_table.apply(
    lambda r: "sofi" in str(r.get("stadium", "")).lower(), axis=1)
if sofi_mask.any():
    sofi_row = fd_table[sofi_mask].index[0]
    print(f"\n  Sofi Stadium → output row {sofi_row}")
    print(f"  FD values: {fd_table.iloc[sofi_row].to_dict()}")
    absorbed = sub_map[sub_map["output_row_number"] == sofi_row]
    print("  Absorbed source rows:")
    for _, r in absorbed.iterrows():
        print(f"    {r['source_table']}  row {r['source_row']}")
    providers = cell_map[cell_map["row_number"] == sofi_row]
    print("  Cell providers (non-null cells only):")
    for _, r in providers.iterrows():
        print(f"    col={r['column_name']}  from {r['source_table']} row {r['source_row']}")

# Ford Field — complemented from t2 (Opened) and t4 (Capacity)
ford_mask = fd_table.apply(
    lambda r: "ford" in str(r.get("stadium", "")).lower(), axis=1)
if ford_mask.any():
    ford_row = fd_table[ford_mask].index[0]
    print(f"\n  Ford Field → output row {ford_row}")
    print(f"  FD values: {fd_table.iloc[ford_row].to_dict()}")
    absorbed = sub_map[sub_map["output_row_number"] == ford_row]
    print("  Absorbed source rows:")
    for _, r in absorbed.iterrows():
        print(f"    {r['source_table']}  row {r['source_row']}")
    providers = cell_map[cell_map["row_number"] == ford_row]
    print("  Cell providers:")
    for _, r in providers.iterrows():
        print(f"    col={r['column_name']}  from {r['source_table']} row {r['source_row']}")

print("\nDone.")
