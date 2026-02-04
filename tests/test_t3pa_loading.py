#!/usr/bin/env python3
"""
Test script to diagnose t3pa file loading issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from radiotrav.io import load_data_as_arrays, detect_file_type


def test_t3pa_loading():
    """Test loading a t3pa file and check coordinate extraction."""
    
    # Create a sample t3pa file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.t3pa', delete=False) as f:
        test_file = f.name
        
        # Write header
        f.write("ToA FToA ToT Matrix Index\n")
        
        # Write some sample data
        # Matrix Index = y * 256 + x
        # So for x=10, y=20: Matrix Index = 20*256 + 10 = 5130
        # For x=50, y=100: Matrix Index = 100*256 + 50 = 25650
        
        test_cases = [
            (1000, 5, 100, 10, 20),  # ToA, FToA, ToT, x, y -> Matrix Index = 20*256+10 = 5130
            (2000, 10, 200, 50, 100),  # Matrix Index = 100*256+50 = 25650
            (3000, 15, 300, 150, 200),  # Matrix Index = 200*256+150 = 51350
        ]
        
        for toa, ftoa, tot, x, y in test_cases:
            matrix_idx = y * 256 + x
            f.write(f"{toa} {ftoa} {tot} {matrix_idx}\n")
        
        print(f"Created test file: {test_file}")
    
    try:
        # Test loading
        print("\n=== Testing load_data_as_arrays ===")
        t, x, y, tot = load_data_as_arrays(test_file, 1.0, None, 0, None)
        
        print(f"Loaded {len(t)} events")
        print(f"Time range: {t.min():.2f} to {t.max():.2f}")
        print(f"X range: {x.min()} to {x.max()}")
        print(f"Y range: {y.min()} to {y.max()}")
        print(f"ToT range: {tot.min():.2f} to {tot.max():.2f}")
        
        print("\n=== First 5 events ===")
        for i in range(min(5, len(t))):
            print(f"Event {i}: t={t[i]:.2f}, x={x[i]}, y={y[i]}, tot={tot[i]:.2f}")
        
        # Verify coordinates
        print("\n=== Verifying coordinate extraction ===")
        expected_coords = [(10, 20), (50, 100), (150, 200)]
        for i, (exp_x, exp_y) in enumerate(expected_coords):
            if i < len(x):
                actual_x, actual_y = x[i], y[i]
                if actual_x == exp_x and actual_y == exp_y:
                    print(f"✓ Event {i}: x={actual_x}, y={actual_y} (correct)")
                else:
                    print(f"✗ Event {i}: expected x={exp_x}, y={exp_y}, got x={actual_x}, y={actual_y}")
        
        # Check if data is sorted
        print("\n=== Checking sort order ===")
        is_sorted = np.all(t[:-1] <= t[1:])
        print(f"Data is sorted by time: {is_sorted}")
        if not is_sorted:
            print("WARNING: Data is not sorted!")
        
    finally:
        # Clean up
        Path(test_file).unlink()


if __name__ == "__main__":
    test_t3pa_loading()
