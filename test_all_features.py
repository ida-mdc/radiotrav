#!/usr/bin/env python3
"""
Comprehensive test script for radiotrap CLI features.

Usage:
    python test_all_features.py <input_file> [--dry-run]

Options:
    --dry-run    Print commands without executing them

This script tests:
1. Process command (full pipeline)
2. Render command (all view modes and display modes)
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil

# Global flag for dry-run mode
DRY_RUN = True


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    
    # Print command in a copy-paste friendly format
    cmd_str = ' '.join(cmd)
    print(f"Command:")
    print(f"  {cmd_str}")
    print()
    
    if DRY_RUN:
        print("[DRY RUN] Command would be executed here")
        print(f"✓ [DRY RUN] {description}")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"✓ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {description}")
        print(f"Exception: {e}")
        return False


def main():
    global DRY_RUN
    
    if len(sys.argv) < 2:
        print("Usage: python test_all_features.py <input_file> [--dry-run]")
        print("\nExample:")
        print("  python test_all_features.py data/events.t3pa")
        print("  python test_all_features.py data/events.t3pa --dry-run")
        sys.exit(1)
    
    # Check for dry-run flag
    args = sys.argv[1:]
    if "--dry-run" in args:
        DRY_RUN = True
        args.remove("--dry-run")
        print("\n[DRY RUN MODE] Commands will be printed but not executed\n")
    
    input_file = args[0]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create test output directory
    test_dir = Path("test_output")
    if not DRY_RUN:
        if test_dir.exists():
            print(f"Cleaning up existing test directory: {test_dir}")
            shutil.rmtree(test_dir)
        test_dir.mkdir()
    else:
        print(f"[DRY RUN] Would create/clean test directory: {test_dir}")
    
    print(f"\n{'='*70}")
    print("RADIOTRAP FEATURE TEST SUITE")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {test_dir}")
    if DRY_RUN:
        print("[DRY RUN MODE - Commands will be printed but not executed]")
    print()
    
    results = []
    
    # ========================================================================
    # TEST 1: Process command (full pipeline)
    # ========================================================================
    process_dir = test_dir / "process_full"
    cmd = [
        "radiotrap", "process",
        input_file,
        str(process_dir),
        "--time-window", "1e6",  # 1ms in nanoseconds
        "--spatial-radius", "1",
        "--sequence-radius", "15.0",
        "--sequence-time-window", "2e10",  # 1 second
    ]
    results.append(("Process (full pipeline)", run_command(cmd, "Full pipeline: segmentation + classification + sequences")))
    
    # ========================================================================
    # TEST 2: Process command (skip existing segmentation)
    # ========================================================================
    if not DRY_RUN and (process_dir / "segmented.txt").exists():
        process_skip_seg_dir = test_dir / "process_skip_segmentation"
        process_skip_seg_dir.mkdir(exist_ok=True)
        # Copy segmented.txt to the new directory so it can be skipped
        shutil.copy(process_dir / "segmented.txt", process_skip_seg_dir / "segmented.txt")
        cmd = [
            "radiotrap", "process",
            input_file,
            str(process_skip_seg_dir),
            "--skip-existing-segmentation",
        ]
        results.append(("Process (skip segmentation)", run_command(cmd, "Skip segmentation using existing segmented.txt")))
    elif DRY_RUN:
        # In dry-run, assume the file would exist and show the command
        process_skip_seg_dir = test_dir / "process_skip_segmentation"
        cmd = [
            "radiotrap", "process",
            input_file,
            str(process_skip_seg_dir),
            "--skip-existing-segmentation",
        ]
        results.append(("Process (skip segmentation)", run_command(cmd, "Skip segmentation using existing segmented.txt")))
    
    # ========================================================================
    # TEST 4: Render - Animated, Classification mode (point to project output dir)
    # ========================================================================
    if DRY_RUN or (process_dir / "classification.csv").exists():
        cmd = [
            "radiotrap", "render",
            str(process_dir),  # project output dir: uses segmented.txt + classification.csv
            str(test_dir / "render_animated_classification.mp4"),
            "--view", "animated",
            "--mode", "classification",
            "--fps", "30",
        ]
        results.append(("Render (animated, classification)", run_command(cmd, "Animated MP4 from project dir (no explicit files)")))
    
    # ========================================================================
    # TEST 4b: Render with radiation filter (match webviewer)
    # ========================================================================
    if DRY_RUN or (process_dir / "classification.csv").exists():
        cmd = [
            "radiotrap", "render",
            str(process_dir),
            str(test_dir / "render_alpha_only.mp4"),
            "--radiation", "Alpha",
            "--view", "animated",
            "--mode", "classification",
            "--fps", "30",
        ]
        results.append(("Render (radiation=Alpha filter)", run_command(cmd, "Animated MP4 with radiation filter Alpha only")))
    
    # ========================================================================
    # TEST 5: Render - Max projection, Classification mode
    # ========================================================================
    if DRY_RUN or (process_dir / "classification.csv").exists():
        cmd = [
            "radiotrap", "render",
            str(process_dir),
            str(test_dir / "render_max_classification.png"),
            "--view", "max",
            "--mode", "classification",
        ]
        results.append(("Render (max, classification)", run_command(cmd, "Max projection PNG with classification colors")))
    
    # ========================================================================
    # TEST 6: Render - Animated, Energy mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_animated_energy.mp4"),
        "--view", "animated",
        "--mode", "energy",
        "--time-window", "0.001",
        "--fps", "30",
    ]
    results.append(("Render (animated, energy)", run_command(cmd, "Animated MP4 with energy (ToT) coloring")))
    
    # ========================================================================
    # TEST 7: Render - Max projection, Energy mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_max_energy.png"),
        "--view", "max",
        "--mode", "energy",
    ]
    results.append(("Render (max, energy)", run_command(cmd, "Max projection PNG with energy coloring")))
    
    # ========================================================================
    # TEST 8: Render - Animated, Density mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_animated_density.mp4"),
        "--view", "animated",
        "--mode", "density",
        "--time-window", "0.001",
        "--fps", "30",
    ]
    results.append(("Render (animated, density)", run_command(cmd, "Animated MP4 with density coloring")))
    
    # ========================================================================
    # TEST 9: Render - Max projection, Density mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_max_density.png"),
        "--view", "max",
        "--mode", "density",
    ]
    results.append(("Render (max, density)", run_command(cmd, "Max projection PNG with density coloring")))
    
    # ========================================================================
    # TEST 10: Render - Animated, Time mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_animated_time.mp4"),
        "--view", "animated",
        "--mode", "time",
        "--time-window", "0.001",
        "--fps", "30",
    ]
    results.append(("Render (animated, time)", run_command(cmd, "Animated MP4 with time coloring")))
    
    # ========================================================================
    # TEST 11: Render - Max projection, Time mode
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_max_time.png"),
        "--view", "max",
        "--mode", "time",
    ]
    results.append(("Render (max, time)", run_command(cmd, "Max projection PNG with time coloring")))
    
    # ========================================================================
    # TEST 12: Render with custom speed
    # ========================================================================
    if DRY_RUN or (process_dir / "classification.csv").exists():
        cmd = [
            "radiotrap", "render",
            str(process_dir),
            str(test_dir / "render_custom_speed.mp4"),
            "--view", "animated",
            "--mode", "classification",
            "--speed", "2.0",
            "--fps", "30",
        ]
        results.append(("Render (custom speed)", run_command(cmd, "Animated MP4 with 2x speed")))
    
    # ========================================================================
    # TEST 13: Render with row filtering
    # ========================================================================
    cmd = [
        "radiotrap", "render",
        input_file,
        str(test_dir / "render_filtered.mp4"),
        "--view", "animated",
        "--mode", "energy",
        "--start-row", "0",
        "--n-rows", "10000",
        "--time-window", "0.001",
        "--fps", "30",
    ]
    results.append(("Render (row filtering)", run_command(cmd, "Animated MP4 with first 10000 rows only")))
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

