#!/usr/bin/env python3
"""
Test script to diagnose load_cluster_ids_chunk issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import the function we're testing
import sys
from pathlib import Path
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from radiotrap.main import load_cluster_ids_chunk, _run_segment


def test_load_cluster_ids():
    """Test loading cluster IDs from a segmented.txt file."""
    
    # Create a temporary segmented.txt file similar to what _run_segment produces
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "segmented.txt"
        
        # Create sample data matching the format from _run_segment
        n_events = 1000
        t = np.random.rand(n_events) * 1e9
        x = np.random.randint(0, 256, n_events)
        y = np.random.randint(0, 256, n_events)
        tot = np.random.rand(n_events) * 1000
        cluster_ids = np.random.randint(0, 50, n_events)
        
        df_out = pd.DataFrame({
            "arrival_time": t,
            "x_pos": x,
            "y_pos": y,
            "ToT": tot,
            "Cluster_ID": cluster_ids,
            # Embedded segmentation metadata
            "seg_time_window_ns": 50.0,
            "seg_spatial_radius_px": 10,
            "seg_input_file": "/path/to/file with spaces/test.t3pa",  # Path with spaces like the real file
            "seg_start_row": 0,
            "seg_n_rows": -1,
        })
        
        print(f"Writing test file to {test_file}")
        df_out.to_csv(test_file, sep=" ", index=False)
        
        # Check what the file looks like
        print("\n=== First 5 lines of file ===")
        with open(test_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    print(f"Line {i}: {line[:100]}")
                else:
                    break
        
        # Try to read header
        print("\n=== Reading header ===")
        try:
            header_df = pd.read_csv(test_file, sep=r"\s+", nrows=0, engine="python")
            print(f"Columns: {list(header_df.columns)}")
            print(f"Column count: {len(header_df.columns)}")
        except Exception as e:
            print(f"Error reading header: {e}")
            return
        
        # Try to read first few rows normally
        print("\n=== Reading first 5 rows normally ===")
        try:
            df_sample = pd.read_csv(test_file, sep=r"\s+", nrows=5, engine="python")
            print(f"Shape: {df_sample.shape}")
            print(f"Columns: {list(df_sample.columns)}")
            print(f"First row Cluster_ID: {df_sample['Cluster_ID'].iloc[0] if 'Cluster_ID' in df_sample.columns else 'NOT FOUND'}")
        except Exception as e:
            print(f"Error reading sample: {e}")
            import traceback
            traceback.print_exc()
        
        # Try load_cluster_ids_chunk with start_row=0, n_rows=None
        print("\n=== Testing load_cluster_ids_chunk(start_row=0, n_rows=None) ===")
        try:
            result = load_cluster_ids_chunk(str(test_file), start_row=0, n_rows=None)
            print(f"Success! Loaded {len(result)} cluster IDs")
            print(f"First 10: {result[:10]}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Try load_cluster_ids_chunk with start_row=0, n_rows=100
        print("\n=== Testing load_cluster_ids_chunk(start_row=0, n_rows=100) ===")
        try:
            result = load_cluster_ids_chunk(str(test_file), start_row=0, n_rows=100)
            print(f"Success! Loaded {len(result)} cluster IDs")
            print(f"First 10: {result[:10]}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Check actual file structure
        print("\n=== File structure analysis ===")
        with open(test_file, 'r') as f:
            lines = f.readlines()
            print(f"Total lines: {len(lines)}")
            print(f"Line 0 (header): {lines[0][:200]}")
            print(f"Line 1 (first data): {lines[1][:200] if len(lines) > 1 else 'N/A'}")
            
            # Count fields in header vs data
            header_fields = len(lines[0].split())
            if len(lines) > 1:
                data_fields = len(lines[1].split())
                print(f"Header fields: {header_fields}, First data row fields: {data_fields}")
                if header_fields != data_fields:
                    print(f"WARNING: Mismatch! Header has {header_fields} fields, data has {data_fields}")


if __name__ == "__main__":
    test_load_cluster_ids()
