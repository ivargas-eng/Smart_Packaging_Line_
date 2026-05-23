"""
Extract DOE data from Excel workbook to CSV.

Usage: python src/extract_doe_data.py --input <path/to/DOE.xlsx>
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def extract(input_path: str, output_path: str = "data/doe_data.csv") -> None:
    """Extract kit data from the 'Runs Z score' sheet."""
    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path, sheet_name="Runs Z score", header=1)

    # Keep relevant columns
    cols = {
        "run_id": "run_id",
        "execution_order": "exec_order",
        "recipe": "recipe",
        "condition": "condition",
        "missing_component": "missing",
        "kit_id": "kit_id",
        "measured_weight_g": "weight",
    }
    df = df[list(cols.keys())].rename(columns=cols)
    df = df.dropna(subset=["weight"])
    df["weight"] = df["weight"].astype(float)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} kits to {output_path}")
    print(f"  OK: {(df['condition']=='OK').sum()}, "
          f"NG: {(df['condition']=='NG').sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to DOE Excel file")
    parser.add_argument("--output", default="data/doe_data.csv")
    args = parser.parse_args()
    extract(args.input, args.output)
