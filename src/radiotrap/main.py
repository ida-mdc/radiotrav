import click
import numpy as np
import pandas as pd
import os
from pathlib import Path
import imageio
from radiotrap.cluster import (
    classify_clusters,
)
from radiotrap.io import load_data_as_arrays, detect_file_type
from radiotrap.render import (
    render_tiff_stream,
    render_video_stream,
    render_yt_discrete,
    render_xy_discrete,
)
from radiotrap.segmentation import segment_events_spatiotemporal
from radiotrap.sequences import find_sequences


# ============================================================
# HELPER: Resolve input (file or project output directory)
# ============================================================
def resolve_render_input(input_path, classification_csv_arg, chains_path_arg):
    """
    If input_path is a directory (project output), use segmented.txt, classification.csv, chains.csv inside.
    Returns (events_path, classification_csv_path, chains_path).
    """
    path = Path(input_path)
    if path.is_dir():
        events_path = path / "segmented.txt"
        class_path = path / "classification.csv"
        chains_path = path / "chains.csv"
        if not events_path.exists():
            raise click.ClickException(
                f"Directory {path} has no segmented.txt. Run 'radiotrap process' first or pass a file."
            )
        return (
            str(events_path),
            str(class_path) if class_path.exists() else (classification_csv_arg or None),
            str(chains_path) if chains_path.exists() else (chains_path_arg or None),
        )
    return (
        str(path),
        classification_csv_arg,
        chains_path_arg,
    )


# ============================================================
# HELPER: Filters (match webviewer)
# ============================================================
def _infer_class(row):
    """Infer class/radiation field from classification row."""
    for col in ("class", "Class", "radiation_type", "Radiation_Type", "radiation", "Radiation", "type", "Type"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return None


def _get_allowed_cluster_ids_classification(df_class, radiation):
    """Set of Cluster_IDs passing radiation filter, or None if ALL (no filter)."""
    if radiation is None or str(radiation).strip().upper() == "ALL":
        return None
    rad = str(radiation).strip()
    df_class = df_class.copy()
    df_class["_class"] = df_class.apply(_infer_class, axis=1)
    filtered = df_class[df_class["_class"].notna() & (df_class["_class"].str.upper() == rad.upper())]
    if "Cluster_ID" in filtered.columns:
        ids = filtered["Cluster_ID"].dropna()
    else:
        ids = filtered.index
    return set(ids.astype(int).astype(str))


def _chain_signature(chain_df):
    """Build signature string from chain rows (Class joined by ' - ')."""
    cls_col = "Class" if "Class" in chain_df.columns else "class"
    if cls_col not in chain_df.columns:
        return ""
    classes = chain_df.sort_values("Sequence_Index")[cls_col].astype(str)
    return " - ".join(classes)


def _chain_passes_constraints(chain_df, max_dt_sec, max_dist_px):
    """True if every step has Time_Delta <= max_dt and Dist_Delta <= max_dist (when set)."""
    if max_dt_sec is None and max_dist_px is None:
        return True
    dt_col = "Time_Delta" if "Time_Delta" in chain_df.columns else "time_delta"
    dd_col = "Dist_Delta" if "Dist_Delta" in chain_df.columns else "dist_delta"
    for _, row in chain_df.iterrows():
        if max_dt_sec is not None and dt_col in row and pd.notna(row[dt_col]):
            if float(row[dt_col]) > max_dt_sec:
                return False
        if max_dist_px is not None and dd_col in row and pd.notna(row[dd_col]):
            if float(row[dd_col]) > max_dist_px:
                return False
    return True


def _get_sequence_filtered_chains(df_chains, query, exact, max_dt_sec, max_dist_px):
    """List of (chain_id, cluster_ids) for chains passing query and constraints."""
    if df_chains is None or df_chains.empty:
        return []
    q = (query or "").strip()
    has_constraints = max_dt_sec is not None or max_dist_px is not None
    if not q and not has_constraints:
        return None  # no filter
    cid_col = "Chain_ID" if "Chain_ID" in df_chains.columns else "chain_id"
    if cid_col not in df_chains.columns:
        return []
    result = []
    q_lower = q.lower()
    for chain_id, grp in df_chains.groupby(cid_col):
        sig = _chain_signature(grp)
        if not _chain_passes_constraints(grp, max_dt_sec, max_dist_px):
            continue
        sig_lower = sig.lower()
        if exact:
            if q_lower and sig_lower != q_lower:
                continue
        else:
            if q_lower and q_lower not in sig_lower:
                continue
        cluster_col = "Cluster_ID" if "Cluster_ID" in grp.columns else "cluster_id"
        if cluster_col in grp.columns:
            cids = set(grp[cluster_col].dropna().astype(int).astype(str))
            result.append((chain_id, cids))
    return result


def _get_allowed_cluster_ids_sequences(chains_list):
    """Set of all Cluster_IDs appearing in chains_list."""
    if chains_list is None:
        return None
    allowed = set()
    for _, cids in chains_list:
        allowed |= cids
    return allowed


def compute_allowed_cluster_ids(classification_csv_path, chains_csv_path, radiation, seq_query, seq_exact, seq_max_dt, seq_max_dist):
    """
    Compute set of allowed Cluster_IDs from classification + sequence filters (match webviewer).
    Returns None if no filters (allow all), else set of cluster id strings.
    """
    cls_allowed = None
    if classification_csv_path and os.path.exists(classification_csv_path):
        df_class = pd.read_csv(classification_csv_path)
        cls_allowed = _get_allowed_cluster_ids_classification(df_class, radiation)

    seq_allowed = None
    if chains_csv_path and os.path.exists(chains_csv_path):
        df_chains = pd.read_csv(chains_csv_path)
        chains_list = _get_sequence_filtered_chains(
            df_chains, seq_query, seq_exact, seq_max_dt, seq_max_dist
        )
        seq_allowed = _get_allowed_cluster_ids_sequences(chains_list)

    if cls_allowed is not None and seq_allowed is not None:
        return cls_allowed & seq_allowed
    if cls_allowed is not None:
        return cls_allowed
    if seq_allowed is not None:
        return seq_allowed
    return None


# ============================================================
# HELPER: Load Cluster IDs Correctly with Offsets
# ============================================================
def load_cluster_ids_chunk(path, start_row, n_rows):
    """
    Helper to load JUST the Cluster_ID column for a specific row chunk.
    Uses CSV reader to handle quoted fields correctly.
    """
    import csv
    
    cluster_ids_list = []
    times_list = []
    
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        header = next(reader)
        
        # Find column indices
        cluster_idx = None
        time_idx = None
        for i, col in enumerate(header):
            clean = col.replace('"', '').replace("'", "").replace(".", "_").strip()
            if clean == "Cluster_ID":
                cluster_idx = i
            if clean in ["arrival_time", "time"]:
                time_idx = i
        
        if cluster_idx is None or time_idx is None:
            raise ValueError(f"Missing required columns. Found: {header}")
        
        # Skip to start_row
        for _ in range(start_row):
            try:
                next(reader)
            except StopIteration:
                break
        
        # Read n_rows
        count = 0
        for row in reader:
            if n_rows is not None and count >= n_rows:
                break
            if len(row) > max(cluster_idx, time_idx):
                try:
                    cluster_id = int(float(row[cluster_idx]))
                    time_val = float(row[time_idx])
                    cluster_ids_list.append(cluster_id)
                    times_list.append(time_val)
                    count += 1
                except (ValueError, IndexError):
                    continue
    
    # Convert to arrays and sort by time (matching load_data_as_arrays)
    cluster_ids_arr = np.array(cluster_ids_list, dtype=np.int32)
    times_arr = np.array(times_list)
    sort_idx = np.argsort(times_arr)
    cluster_ids = cluster_ids_arr[sort_idx]
    
    return cluster_ids


# ============================================================
# CLI
# ============================================================

@click.group()
def cli():
    pass


def _run_segment(input_file, output_txt, time_window, spatial_radius, start_row, n_rows):
    """
    Core implementation for segmentation (no Click dependencies).
    Used by the `process` pipeline.
    """
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, None, start_row, n_rows)

    click.echo(f"Segmenting {len(t)} events...")
    cluster_ids = segment_events_spatiotemporal(t, x, y, time_window, spatial_radius)

    # Save, embedding segmentation parameters as metadata columns (constant per row)
    click.echo(f"Saving to {output_txt}...")
    df_out = pd.DataFrame({
        "arrival_time": t,
        "x_pos": x,
        "y_pos": y,
        "ToT": tot,
        "Cluster_ID": cluster_ids,
        # Embedded segmentation metadata
        "seg_time_window_ns": float(time_window),
        "seg_spatial_radius_px": int(spatial_radius),
        "seg_input_file": input_file,
        "seg_start_row": int(start_row),
        "seg_n_rows": -1 if n_rows is None else int(n_rows),
    })
    df_out.to_csv(output_txt, sep=" ", index=False)
    click.echo("Done.")


def _run_classify(input_file, output_csv, start_row, n_rows):
    """
    Core implementation for classification (no Click dependencies).
    Used by the `process` pipeline.
    """
    file_type = detect_file_type(input_file)
    if file_type != "txt":
        raise click.ClickException(
            "Classification requires a TXT file containing a 'Cluster_ID' column "
            "(i.e. the output of 'radiotrap process'). "
            f"Got: {input_file}\n"
            "Run:\n"
            "  radiotrap process <input.t3pa|input.txt> output_dir --time-window <...>\n"
            "This will generate segmented.txt, classification.csv, and chains.csv."
        )

    # Load data normally (this sorts by time)
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, None, start_row, n_rows)

    # Load cluster IDs separately - need to match the sorting from load_data_as_arrays
    # The issue is that quoted paths cause column misalignment with sep=r"\s+"
    # Solution: Read file with CSV reader to handle quoted fields correctly
    import csv
    
    cluster_ids_list = []
    times_list = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        header = next(reader)
        
        # Find column indices
        cluster_idx = None
        time_idx = None
        for i, col in enumerate(header):
            clean = col.replace('"', '').replace("'", "").replace(".", "_").strip()
            if clean == "Cluster_ID":
                cluster_idx = i
            if clean in ["arrival_time", "time"]:
                time_idx = i
        
        if cluster_idx is None or time_idx is None:
            raise ValueError(f"Missing required columns. Found: {header}")
        
        # Skip to start_row
        for _ in range(start_row):
            try:
                next(reader)
            except StopIteration:
                break
        
        # Read n_rows
        count = 0
        for row in reader:
            if n_rows is not None and count >= n_rows:
                break
            if len(row) > max(cluster_idx, time_idx):
                try:
                    cluster_id = int(float(row[cluster_idx]))
                    time_val = float(row[time_idx])
                    cluster_ids_list.append(cluster_id)
                    times_list.append(time_val)
                    count += 1
                except (ValueError, IndexError):
                    continue
    
    # Convert to arrays and sort by time (matching load_data_as_arrays)
    cluster_ids_arr = np.array(cluster_ids_list, dtype=np.int32)
    times_arr = np.array(times_list)
    sort_idx = np.argsort(times_arr)
    cluster_ids = cluster_ids_arr[sort_idx]

    # Reconstruct DF for calculation
    df_calc = pd.DataFrame({
        "x_pos": x,
        "y_pos": y,
        "ToT": tot,
        "arrival_time": t,
        "Cluster_ID": cluster_ids
    })

    # Run
    stats = classify_clusters(df_calc)

    # ------------------------------------------------------------------
    # Embed segmentation + classification context as metadata columns.
    # These are constant per row and safe for all downstream tools.
    # ------------------------------------------------------------------
    meta_values = {}

    # 1) Try to propagate segmentation metadata from the segmented TXT, if present.
    # Read the full file to get metadata columns (they're constant per row anyway)
    try:
        # Read just the first few rows to get metadata (faster than full file)
        meta_df = pd.read_csv(input_file, sep=r"\s+", nrows=10, engine="python")
        # Normalize column names (remove quotes, dots, etc.) to match what segment writes
        meta_df.columns = meta_df.columns.str.replace('"', '').str.replace("'", "").str.replace(".", "_").str.strip()
    except Exception as e:
        click.echo(f"Warning: Could not read segmentation metadata: {e}")
        meta_df = None

    if meta_df is not None and len(meta_df) > 0:
        for col in [
            "seg_time_window_ns",
            "seg_spatial_radius_px",
            "seg_input_file",
            "seg_start_row",
            "seg_n_rows",
        ]:
            if col in meta_df.columns:
                # Get the first non-null value (should be the same for all rows)
                val = meta_df[col].dropna()
                if len(val) > 0:
                    meta_values[col] = val.iloc[0]

    # 2) Add classification call context
    meta_values["classify_input_file"] = input_file
    meta_values["classify_start_row"] = int(start_row)
    meta_values["classify_n_rows"] = -1 if n_rows is None else int(n_rows)

    # Add all metadata columns to stats DataFrame (constant for all rows)
    for k, v in meta_values.items():
        stats[k] = v

    click.echo(f"Saving classification summary to {output_csv}...")
    stats.to_csv(output_csv, index=True)
    click.echo("Done.")


def _run_analyze_sequences(input_csv, output_chains_csv, radius, time_window, keep_noise, max_chain_length=None, pattern_lookup_file=None):
    """
    Core implementation for sequence analysis (no Click dependencies).
    Used by the `process` pipeline.
    """
    click.echo(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Run new stitching logic
    chains_df = find_sequences(
        df, 
        spatial_radius=radius, 
        max_time_gap=time_window, 
        exclude_noise=not keep_noise,
        max_chain_length=max_chain_length,
        pattern_lookup_file=pattern_lookup_file
    )

    if len(chains_df) == 0:
        click.echo("No chains found.")
        return

    # Count unique chains
    n_chains = chains_df["Chain_ID"].nunique()
    click.echo(f"Found {n_chains} chains involving {len(chains_df)} events.")

    # Add sequence processing metadata as constant columns (same for all rows)
    chains_df["seq_spatial_radius_px"] = float(radius)
    chains_df["seq_time_window_ns"] = float(time_window)
    chains_df["seq_max_chain_length"] = -1 if max_chain_length is None else int(max_chain_length)
    chains_df["seq_pattern_lookup_file"] = "" if pattern_lookup_file is None else str(pattern_lookup_file)
    chains_df["seq_input_file"] = input_csv

    chains_df.to_csv(output_chains_csv, index=False)
    click.echo(f"Saved -> {output_chains_csv}")


# --- RENDER (Unified, matching webviewer options) ---
@cli.command()
@click.argument("input_path")
@click.argument("output_file")
@click.option("--classification-csv", type=click.Path(exists=True), default=None, help="Classification CSV (optional if input is project output dir with classification.csv).")
@click.option("--chains-csv", type=click.Path(exists=True), default=None, help="Chains CSV for sequence filter (optional if input is project output dir with chains.csv).")
@click.option("--radiation", type=click.Choice(["ALL", "Alpha", "Beta", "Gamma", "Other"]), default="ALL", help="Filter by radiation type (match webviewer).")
@click.option("--seq-query", default=None, help="Filter chains by signature (e.g. 'Alpha - Beta'). Match webviewer sequence query.")
@click.option("--seq-exact", is_flag=True, help="Match chain signature exactly (otherwise contains).")
@click.option("--seq-max-dt", type=float, default=None, help="Max time delta in chain (seconds). Filter chains by step duration.")
@click.option("--seq-max-dist", type=float, default=None, help="Max spatial distance in chain (px). Filter chains by step distance.")
@click.option("--view", type=click.Choice(["animated", "max"]), default="animated", help="View mode: animated (mp4) or max projection (png).")
@click.option("--projection", type=click.Choice(["xy", "yt"]), default="xy", help="Projection type: xy (top-down) or yt (side view, Y vs Time).")
@click.option("--mode", type=click.Choice(["classification", "energy"]), default="classification", help="Display mode.")
@click.option("--bin-size", type=float, default=None, help="Time bin size in nanoseconds (e.g., 10000000 for 10ms). If not set, derived from --speed and --fps.")
@click.option("--time-window", type=float, default=None, help="Fade in/out window (nanoseconds, e.g. 1e9 for 1s). Events within this window around each frame center are blended with alpha fade; does not affect the number of frames.")
@click.option("--speed", type=float, default=1.0, help="Playback speed (× real-time). Higher = fewer frames, faster render. Only used when --bin-size is not set.")
@click.option("--fps", default=30, help="Frames per second (for mp4 format).")
@click.option("--start-row", default=0, help="Row index to start.")
@click.option("--n-rows", default=None, type=int, help="Number of rows to process.")
def render(input_path, output_file, classification_csv, chains_csv, radiation, seq_query, seq_exact, seq_max_dt, seq_max_dist, view, mode, projection, bin_size, time_window, speed, fps, start_row, n_rows):
    """
    Render events to video/image with options matching the webviewer.
    
    INPUT_PATH can be a project output directory (with segmented.txt, classification.csv, chains.csv)
    or a single events file (segmented.txt or raw). Filters match the webviewer (radiation, sequence query).
    
    - View: animated (mp4 video) or max (png image)
    - Projection: xy (top-down) or yt (side view, Y vs Time)
    - Mode: classification or energy
    """
    
    # Resolve input: directory -> segmented.txt + classification.csv + chains.csv
    events_file, classification_csv_path, chains_path = resolve_render_input(input_path, classification_csv, chains_csv)
    
    # YT projection only works with animated view
    if projection == "yt" and view == "max":
        raise click.ClickException("YT projection is only available for animated view, not max projection.")
    
    # Determine output format from view mode
    if view == "max":
        # Max projection -> PNG
        if not output_file.lower().endswith('.png'):
            output_file = output_file.rsplit('.', 1)[0] + '.png'
    else:
        # Animated -> MP4
        if not output_file.lower().endswith('.mp4'):
            output_file = output_file.rsplit('.', 1)[0] + '.mp4'
    
    # Load events
    t, x, y, tot = load_data_as_arrays(events_file, 1.0, None, start_row, n_rows)
    
    # Apply filters (match webviewer): load cluster IDs if segmented, then mask by radiation/sequence
    if detect_file_type(events_file) == "txt":
        cluster_ids_full = load_cluster_ids_chunk(events_file, start_row, n_rows)
        allowed_ids = compute_allowed_cluster_ids(
            classification_csv_path, chains_path,
            radiation, seq_query, seq_exact, seq_max_dt, seq_max_dist,
        )
        if allowed_ids is not None:
            mask = np.array([str(cid) in allowed_ids for cid in cluster_ids_full])
            n_before = len(t)
            t, x, y, tot = t[mask], x[mask], y[mask], tot[mask]
            cluster_ids_full = cluster_ids_full[mask]
            click.echo(f"Filter: {len(t)} / {n_before} events (radiation={radiation}, seq-query={seq_query or '-'})")
            if len(t) == 0:
                raise click.ClickException(
                    f"No events match the filter criteria. "
                    f"Try adjusting --radiation, --seq-query, --seq-max-dt, or --seq-max-dist parameters."
                )
        cluster_ids = cluster_ids_full
    else:
        cluster_ids = None
    
    # Calculate time range (time is in nanoseconds)
    t_min = t.min() if len(t) > 0 else 0.0
    t_max = t.max() if len(t) > 0 else 1.0
    t_range = t_max - t_min
    
    # Determine bin_size: --bin-size if set, else derive from speed + fps.
    # time_window is only for fade in/out and must not affect the number of frames.
    # Higher speed = more data time per frame = larger bin = fewer frames = faster render.
    if bin_size is not None:
        bin_size_ns = float(bin_size)  # Already in nanoseconds (speed has no effect)
    else:
        # Derive from speed and fps: each frame advances (speed/fps) seconds of data
        # So at speed=1, fps=30 -> ~33 ms per frame; at speed=10, fps=30 -> ~333 ms per frame (10x fewer frames)
        bin_size_ns = (speed / float(fps)) * 1e9
        bin_size_ns = max(bin_size_ns, 1.0)  # avoid zero
    
    if view == "max":
        # Max projection: create single PNG image
        # Aggregate all events into a 2D image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        if mode == "classification":
            file_type = detect_file_type(events_file)
            if file_type != "txt":
                raise click.ClickException(
                    "Classification mode requires a segmented TXT file with Cluster_ID column. "
                    f"Got: {events_file}"
                )
            
            if classification_csv_path is None:
                raise click.ClickException(
                    "Classification mode requires classification.csv (use project output dir or --classification-csv)."
                )
            
            if cluster_ids is None:
                cluster_ids = load_cluster_ids_chunk(events_file, start_row, n_rows)
            
            # Build color lookup from classification CSV
            palette = {
                "Alpha": np.array([220, 20, 20], dtype=np.uint8),
                "Beta": np.array([20, 180, 220], dtype=np.uint8),
                "Gamma": np.array([20, 220, 20], dtype=np.uint8),
                "Other": np.array([150, 150, 150], dtype=np.uint8)
            }
            
            df_class = pd.read_csv(classification_csv_path)
            max_id = df_class["Cluster_ID"].max()
            if np.isnan(max_id) or max_id < 0:
                max_id = 0
            color_lookup = np.zeros((int(max_id) + 2, 3), dtype=np.uint8)
            for _, row in df_class.iterrows():
                color_lookup[int(row["Cluster_ID"])] = palette.get(row["class"], palette["Other"])
            
            # For max projection, take the highest energy event at each pixel
            pixel_map = {}  # (x, y) -> (max_energy, cluster_id)
            for i in range(len(x)):
                xi, yi = int(x[i]), int(y[i])
                if 0 <= xi < 256 and 0 <= yi < 256:
                    cid = cluster_ids[i]
                    energy = tot[i] if i < len(tot) else 0
                    key = (xi, yi)
                    if key not in pixel_map or energy > pixel_map[key][0]:
                        pixel_map[key] = (energy, cid)
            
            # Fill image with colors from max energy events
            for (xi, yi), (_, cid) in pixel_map.items():
                if 0 <= cid < len(color_lookup):
                    img[yi, xi] = color_lookup[cid]
        
        elif mode == "energy":
            # For energy mode, use max projection of ToT values
            # Aggregate by pixel, taking max value
            pixel_map = {}  # (x, y) -> max_value
            for i in range(len(x)):
                xi, yi = int(x[i]), int(y[i])
                if 0 <= xi < 256 and 0 <= yi < 256:
                    key = (xi, yi)
                    val = tot[i] if i < len(tot) else 0
                    if key not in pixel_map or val > pixel_map[key]:
                        pixel_map[key] = val
            
            # Normalize and apply colormap (turbo-like)
            if pixel_map:
                max_val = max(pixel_map.values())
                min_val = min(pixel_map.values())
                if max_val > min_val:
                    from matplotlib.cm import get_cmap
                    cmap = get_cmap('turbo')
                    for (xi, yi), val in pixel_map.items():
                        normalized = (val - min_val) / (max_val - min_val)
                        rgb = cmap(normalized)[:3]  # Get RGB, ignore alpha
                        img[yi, xi] = (np.array(rgb) * 255).astype(np.uint8)
                else:
                    # All same value, use single color
                    rgb = get_cmap('turbo')(0.5)[:3]
                    for (xi, yi) in pixel_map.keys():
                        img[yi, xi] = (np.array(rgb) * 255).astype(np.uint8)
        
        # Save PNG
        imageio.imsave(output_file, img)
        click.echo(f"Saved max projection → {output_file}")
    
    else:
        # Animated view: create MP4 video
        if projection == "yt":
            # YT projection (side view: Y vs Time, scanning through X)
            # Requires segmented file with Cluster_ID
            file_type = detect_file_type(events_file)
            if file_type != "txt":
                raise click.ClickException(
                    "YT projection requires a segmented TXT file with Cluster_ID column. "
                    f"Got: {events_file}"
                )
            
            if cluster_ids is None:
                cluster_ids = load_cluster_ids_chunk(events_file, start_row, n_rows)
            
            # Build color lookup if classification CSV is provided
            color_lookup = None
            if classification_csv_path is not None:
                palette = {
                    "Alpha": np.array([220, 20, 20], dtype=np.uint8),
                    "Beta": np.array([20, 180, 220], dtype=np.uint8),
                    "Gamma": np.array([20, 220, 20], dtype=np.uint8),
                    "Other": np.array([150, 150, 150], dtype=np.uint8)
                }
                
                df_class = pd.read_csv(classification_csv_path)
                max_id = df_class["Cluster_ID"].max()
                if np.isnan(max_id) or max_id < 0:
                    max_id = 0
                color_lookup = np.zeros((int(max_id) + 2, 3), dtype=np.uint8)
                for _, row in df_class.iterrows():
                    color_lookup[int(row["Cluster_ID"])] = palette.get(row["class"], palette["Other"])
            
            # Sort by time, then by x (required for YT projection)
            time_sort = np.argsort(t)
            t = t[time_sort]
            x = x[time_sort]
            y = y[time_sort]
            cluster_ids = cluster_ids[time_sort]
            
            x_sort = np.argsort(x)
            t = t[x_sort]
            x = x[x_sort]
            y = y[x_sort]
            cluster_ids = cluster_ids[x_sort]
            
            t_limits = (t[0], t[-1])
            render_yt_discrete(t, x, y, cluster_ids, output_file, bin_size_ns, fps, color_lookup=color_lookup, t_limits=t_limits)
        
        else:
            # XY projection (top-down view)
            if mode == "classification":
                file_type = detect_file_type(events_file)
                if file_type != "txt":
                    raise click.ClickException(
                        "Classification mode requires a segmented TXT file with Cluster_ID column. "
                        f"Got: {events_file}"
                    )
                
                if classification_csv_path is None:
                    raise click.ClickException(
                        "Classification mode requires classification.csv (use project output dir or --classification-csv)."
                    )
                
                if cluster_ids is None:
                    cluster_ids = load_cluster_ids_chunk(events_file, start_row, n_rows)
                
                # Build color lookup from classification CSV
                palette = {
                    "Alpha": np.array([220, 20, 20], dtype=np.uint8),
                    "Beta": np.array([20, 180, 220], dtype=np.uint8),
                    "Gamma": np.array([20, 220, 20], dtype=np.uint8),
                    "Other": np.array([150, 150, 150], dtype=np.uint8)
                }
                
                df_class = pd.read_csv(classification_csv_path)
                max_id = df_class["Cluster_ID"].max()
                if np.isnan(max_id) or max_id < 0:
                    max_id = 0
                color_lookup = np.zeros((int(max_id) + 2, 3), dtype=np.uint8)
                for _, row in df_class.iterrows():
                    color_lookup[int(row["Cluster_ID"])] = palette.get(row["class"], palette["Other"])
                
                # Sort by time
                local_sort = np.argsort(t)
                t = t[local_sort]
                x = x[local_sort]
                y = y[local_sort]
                cluster_ids = cluster_ids[local_sort]
                
                # Render with classification colors
                # Adjust time_window if it's smaller than bin_size to ensure events are visible
                effective_time_window = time_window
                if time_window is not None and time_window > 0:
                    min_time_window = bin_size_ns * 1.5
                    if time_window < min_time_window:
                        effective_time_window = min_time_window
                        click.echo(f"Adjusting time_window from {time_window/1e9:.3f}s to {effective_time_window/1e9:.3f}s to match bin_size")
                render_xy_discrete(t, x, y, cluster_ids, output_file, bin_size_ns, fps, color_lookup=color_lookup, time_window_ns=effective_time_window)
            
            elif mode == "energy":
                # For energy mode, render raw events (no classification needed)
                # Adjust time_window if it's smaller than bin_size to ensure events are visible
                # Use at least 1.5x bin_size to ensure good coverage
                effective_time_window = time_window
                if time_window is not None and time_window > 0:
                    min_time_window = bin_size_ns * 1.5
                    if time_window < min_time_window:
                        effective_time_window = min_time_window
                        click.echo(f"Adjusting time_window from {time_window/1e9:.3f}s to {effective_time_window/1e9:.3f}s to match bin_size")
                render_video_stream(t, x, y, tot, output_file, bin_size_ns, fps, time_window_ns=effective_time_window)
            else:
                raise click.ClickException(f"Unsupported mode: {mode}")
    
    click.echo(f"Done → {output_file}")


# --- PROCESS PIPELINE ---
@cli.command()
@click.argument("input_file")
@click.argument("output_dir")
@click.option("--time-window", required=False, type=float, default=None, help="Segmentation time window (ns). Required unless using --skip-existing-segmentation.")
@click.option("--spatial-radius", default=1, type=int, help="Segmentation spatial radius (px). Only used if segmentation is run.")
@click.option("--sequence-radius", default=15.0, type=float, help="Sequence analysis spatial radius (px).")
@click.option(
    "--sequence-time-window",
    default=1e9,
    type=float,
    help="Sequence analysis max time gap between events in a chain (ns).",
)
@click.option("--sequence-max-length", type=int, default=None, help="Maximum number of events in a chain. Longer chains will be ignored.")
@click.option("--sequence-pattern-lookup", type=click.Path(exists=True), default=None, help="CSV file with pattern definitions for sequence matching (columns: pattern_name, pattern_sequence).")
@click.option("--start-row", default=0, help="Row index to start (for all steps).")
@click.option("--n-rows", default=None, type=int, help="Number of rows to process (for all steps).")
@click.option(
    "--skip-existing-segmentation",
    is_flag=True,
    help="If set, skip segmentation if segmented.txt already exists in output directory. Useful when segmentation is expensive and you only want to redo classification/sequences.",
)
@click.option(
    "--skip-existing-classification",
    is_flag=True,
    help="If set, skip classification if classification.csv already exists in output directory. Useful when classification is expensive and you only want to redo sequence analysis.",
)
def process(
    input_file,
    output_dir,
    time_window,
    spatial_radius,
    sequence_radius,
    sequence_time_window,
    sequence_max_length,
    sequence_pattern_lookup,
    start_row,
    n_rows,
    skip_existing_segmentation,
    skip_existing_classification,
):
    """
    Full pipeline:

    1) Segment events (from T3PA or TXT)
    2) Classify clusters
    3) Analyze decay sequences
    4) Copy HTML dashboards into the output folder

    Use --skip-existing-segmentation to skip the expensive segmentation step if
    segmented.txt already exists in the output directory.
    
    Use --skip-existing-classification to skip classification if classification.csv
    already exists. This is useful when you only want to redo sequence analysis.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_txt = out_dir / "segmented.txt"
    class_csv = out_dir / "classification.csv"
    chains_csv = out_dir / "chains.csv"

    click.echo(f"Output folder: {out_dir}")

    # Determine if segmentation should be skipped
    skip_segmentation = skip_existing_segmentation and seg_txt.exists()
    
    # Validate segmentation parameters if needed
    if not skip_segmentation:
        if time_window is None:
            raise click.ClickException("--time-window is required when running segmentation. Use --skip-existing-segmentation to skip if segmented.txt already exists.")

    # 1) SEGMENT
    if skip_segmentation:
        click.echo("=== SKIPPING Segmentation (--skip-existing-segmentation, file exists) ===")
        if not seg_txt.exists():
            raise click.ClickException(
                f"Segmented file not found: {seg_txt}\n"
                "Cannot skip segmentation without segmented.txt. Remove --skip-existing-segmentation or run full pipeline first."
            )
        click.echo(f"Using existing segmented file: {seg_txt}")
    else:
        click.echo("=== STEP 1/3: Segmentation ===")
        _run_segment(
            input_file=input_file,
            output_txt=str(seg_txt),
            time_window=time_window,
            spatial_radius=spatial_radius,
            start_row=start_row,
            n_rows=n_rows,
        )

    # 2) CLASSIFY
    skip_classification = skip_existing_classification and class_csv.exists()
    if skip_classification:
        click.echo("=== SKIPPING Classification (--skip-existing-classification, file exists) ===")
        if not class_csv.exists():
            raise click.ClickException(
                f"Classification file not found: {class_csv}\n"
                "Cannot skip classification without classification.csv. Remove --skip-existing-classification or run full pipeline first."
            )
        click.echo(f"Using existing classification file: {class_csv}")
    else:
        click.echo("=== STEP 2/3: Classification ===")
        _run_classify(
            input_file=str(seg_txt),
            output_csv=str(class_csv),
            start_row=0,
            n_rows=None,
        )

    # 3) SEQUENCE ANALYSIS
    click.echo("=== STEP 3/3: Sequence Analysis ===")
    _run_analyze_sequences(
        input_csv=str(class_csv),
        output_chains_csv=str(chains_csv),
        radius=sequence_radius,
        time_window=sequence_time_window,
        keep_noise=False,
        max_chain_length=sequence_max_length,
        pattern_lookup_file=sequence_pattern_lookup,
    )

    # 4) Copy HTML dashboard into output folder
    click.echo("Copying HTML dashboard into output folder...")
    project_root = Path(__file__).resolve().parents[2]
    report_dir = project_root / "report"
    dashboard_src = report_dir / "dashboard.html"

    try:
        dashboard_dst = out_dir / "dashboard.html"
        with dashboard_src.open("r", encoding="utf-8") as f_in, dashboard_dst.open("w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
        click.echo(f"Dashboard copied to {dashboard_dst}")
    except FileNotFoundError:
        click.echo(f"Warning: Dashboard template not found: {dashboard_src}")

    click.echo("Done. Open dashboard.html in a browser and load the generated CSV files.")

if __name__ == "__main__":
    cli()
