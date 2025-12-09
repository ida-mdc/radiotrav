import click
import numpy as np
import pandas as pd
from radiotrap.cluster import (
    classify_clusters,
)
from radiotrap.io import load_data_as_arrays
from radiotrap.render import render_tiff_stream, render_video_stream, render_yt_discrete, render_xy_discrete
from radiotrap.segmentation import segment_events_spatiotemporal


# ============================================================
# HELPER: Load Cluster IDs Correctly with Offsets
# ============================================================
def load_cluster_ids_chunk(path, start_row, n_rows):
    """
    Helper to load JUST the Cluster_ID column for a specific row chunk.
    """
    # 1. Read header row only to get all column names
    header_df = pd.read_csv(path, sep=r"\s+", nrows=0, engine="python")
    original_columns = list(header_df.columns)

    # 2. Find the exact column name for "Cluster_ID" (handling quotes/spaces)
    cluster_col_name = None
    for col in original_columns:
        clean_col = col.replace('"', '').replace("'", "").replace(".", "_").strip()
        if clean_col == "Cluster_ID":
            cluster_col_name = col
            break

    if cluster_col_name is None:
        raise ValueError(f"Input file missing 'Cluster_ID' column. Found: {original_columns}")

    # 3. Read specific chunk
    # CRITICAL FIX: We must pass 'names=original_columns' so pandas maps the data
    # to the correct column names, even though we skipped the header row.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,  # We are skipping the real header
        names=original_columns,  # So we must provide names manually
        usecols=[cluster_col_name],
        skiprows=start_row + 1,  # +1 to skip the header line itself
        nrows=n_rows,
        dtype=str,
        engine="python"
    )

    return pd.to_numeric(df[cluster_col_name], errors="coerce").fillna(0).astype(np.int32).values


# ============================================================
# CLI
# ============================================================

@click.group()
def cli():
    pass


# --- 1. RENDER RAW ---
@cli.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--bin-size", default=1e3, help="Time bin size.")
@click.option("--fps", default=30)
@click.option("--format", "out_format", type=click.Choice(["mp4", "tiff"]), default="mp4")
@click.option("--start-row", default=0, help="Row index to start.")
@click.option("--n-rows", default=None, type=int, help="Number of rows to process.")
def render(input_file, output_file, bin_size, fps, out_format, start_row, n_rows):
    """
    Render raw events (TXT or T3PA).
    """
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)

    if out_format == "tiff":
        render_tiff_stream(t, x, y, output_file, bin_size)
    else:
        render_video_stream(t, x, y, tot, output_file, bin_size, fps)
    click.echo(f"Done → {output_file}")


# --- 2. SEGMENT ---
@cli.command()
@click.argument("input_file")
@click.argument("output_txt")
@click.option("--time-window", required=True, type=float, help="Max time gap (ns).")
@click.option("--spatial-radius", default=1, type=int, help="Search radius (px).")
@click.option("--start-row", default=0, help="Row index to start.")
@click.option("--n-rows", default=None, type=int, help="Number of rows to process.")
def segment(input_file, output_txt, time_window, spatial_radius, start_row, n_rows):
    """
    Groups neighbors (Time + Space) and saves Cluster_ID.
    """
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)

    click.echo(f"Segmenting {len(t)} events...")
    cluster_ids = segment_events_spatiotemporal(t, x, y, time_window, spatial_radius)

    # Save
    click.echo(f"Saving to {output_txt}...")
    df_out = pd.DataFrame({
        "arrival_time": t,
        "x_pos": x,
        "y_pos": y,
        "ToT": tot,
        "Cluster_ID": cluster_ids
    })
    df_out.to_csv(output_txt, sep=" ", index=False)
    click.echo("Done.")


# --- 3. CLASSIFY ---
@cli.command()
@click.argument("input_file")
@click.argument("output_csv")
@click.option("--start-row", default=0, help="Row index to start.")
@click.option("--n-rows", default=None, type=int, help="Number of rows to process.")
def classify(input_file, output_csv, start_row, n_rows):
    """
    Classifies clusters and saves detailed stats + thumbnails.
    """
    # Load data normally
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)

    # Load IDs separately
    cluster_ids = load_cluster_ids_chunk(input_file, start_row, n_rows)

    # Reconstruct DF for calculation
    # FIX: Added "arrival_time": t
    df_calc = pd.DataFrame({
        "x_pos": x,
        "y_pos": y,
        "ToT": tot,
        "arrival_time": t,
        "Cluster_ID": cluster_ids
    })

    # Run
    stats = classify_clusters(df_calc)

    click.echo(f"Saving classification summary to {output_csv}...")
    stats.to_csv(output_csv, index=True)
    click.echo("Done.")


@cli.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--bin-size", default=1e3)
@click.option("--fps", default=30)
@click.option("--format", "out_format", type=click.Choice(["mp4", "tiff"]), default="mp4")
@click.option("--start-row", default=0)
@click.option("--n-rows", default=None, type=int)
def render(input_file, output_file, bin_size, fps, out_format, start_row, n_rows):
    t, x, y, tot = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)
    if out_format == "tiff":
        render_tiff_stream(t, x, y, output_file, bin_size)
    else:
        render_video_stream(t, x, y, tot, output_file, bin_size, fps)
    click.echo(f"Done → {output_file}")


# --- 4. RENDER SEGMENTATION (Random Colors) ---
@cli.command()
@click.argument("input_file")
@click.argument("output_video")
@click.option("--bin-size", required=True, type=float)
@click.option("--fps", default=30)
@click.option("--start-row", default=0)
@click.option("--n-rows", default=None, type=int)
def render_segmentation(input_file, output_video, bin_size, fps, start_row, n_rows):
    """
    Renders XY video with random colors (Segmentation).
    """
    t, x, y, _ = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)
    cluster_ids = load_cluster_ids_chunk(input_file, start_row, n_rows)

    local_sort = np.argsort(t)
    t = t[local_sort]
    x = x[local_sort]
    y = y[local_sort]
    cluster_ids = cluster_ids[local_sort]

    # CALL UNIFIED FUNCTION (color_lookup=None)
    render_xy_discrete(t, x, y, cluster_ids, output_video, bin_size, fps, color_lookup=None)
    click.echo(f"Saved → {output_video}")


# --- 5. RENDER CLASSIFICATION (Specific Colors) ---
@cli.command()
@click.argument("input_events_txt")
@click.argument("input_class_csv")
@click.argument("output_video")
@click.option("--bin-size", required=True, type=float)
@click.option("--fps", default=30)
@click.option("--start-row", default=0)
@click.option("--n-rows", default=None, type=int)
def render_classification(input_events_txt, input_class_csv, output_video, bin_size, fps, start_row, n_rows):
    """
    Renders XY video with class colors.
    """
    palette = {
        "Alpha": np.array([220, 20, 20], dtype=np.uint8),
        "Beta": np.array([20, 180, 220], dtype=np.uint8),
        "Gamma": np.array([20, 220, 20], dtype=np.uint8),
        "Other": np.array([150, 150, 150], dtype=np.uint8)
    }

    df_class = pd.read_csv(input_class_csv)
    max_id = df_class["Cluster_ID"].max()
    if np.isnan(max_id) or max_id < 0: max_id = 0
    color_lookup = np.zeros((int(max_id) + 2, 3), dtype=np.uint8)
    for _, row in df_class.iterrows():
        color_lookup[int(row["Cluster_ID"])] = palette.get(row["class"], palette["Other"])

    t, x, y, _ = load_data_as_arrays(input_events_txt, 1.0, 'txt', start_row, n_rows)
    cluster_ids = load_cluster_ids_chunk(input_events_txt, start_row, n_rows)

    local_sort = np.argsort(t)
    t = t[local_sort]
    x = x[local_sort]
    y = y[local_sort]
    cluster_ids = cluster_ids[local_sort]

    # CALL UNIFIED FUNCTION (with lookup)
    render_xy_discrete(t, x, y, cluster_ids, output_video, bin_size, fps, color_lookup=color_lookup)
    click.echo(f"Saved → {output_video}")


# --- 6. RENDER YT (Classification) ---
@cli.command()
@click.argument("input_events_txt")
@click.argument("input_class_csv")
@click.argument("output_video")
@click.option("--bin-size", required=True, type=float)
@click.option("--fps", default=30)
@click.option("--start-row", default=0)
@click.option("--n-rows", default=None, type=int)
def render_yt(input_events_txt, input_class_csv, output_video, bin_size, fps, start_row, n_rows):
    """
    Renders YT video with class colors.
    """
    # 1. Build Lookup (Same as above)
    palette = {"Alpha": [220, 20, 20], "Beta": [20, 180, 220], "Gamma": [20, 220, 20], "Other": [150, 150, 150]}
    df_class = pd.read_csv(input_class_csv)
    max_id = df_class["Cluster_ID"].max()
    color_lookup = np.zeros((int(max_id) + 2, 3), dtype=np.uint8)
    for _, row in df_class.iterrows():
        color_lookup[int(row["Cluster_ID"])] = palette.get(row["class"], palette["Other"])

    # 2. Load
    t, x, y, _ = load_data_as_arrays(input_events_txt, 1.0, 'txt', start_row, n_rows)
    cluster_ids = load_cluster_ids_chunk(input_events_txt, start_row, n_rows)

    local_sort = np.argsort(t)
    t, x, y, cluster_ids = t[local_sort], x[local_sort], y[local_sort], cluster_ids[local_sort]
    t_limits = (t[0], t[-1])

    x_sort_idx = np.argsort(x)
    t, x, y, cluster_ids = t[x_sort_idx], x[x_sort_idx], y[x_sort_idx], cluster_ids[x_sort_idx]

    # CALL UNIFIED YT (with lookup)
    render_yt_discrete(t, x, y, cluster_ids, output_video, bin_size, fps, color_lookup=color_lookup, t_limits=t_limits)
    click.echo(f"Saved → {output_video}")


# --- 7. RENDER SEGMENTATION YT (Random) ---
@cli.command()
@click.argument("input_file")
@click.argument("output_video")
@click.option("--bin-size", required=True, type=float)
@click.option("--fps", default=30)
@click.option("--start-row", default=0)
@click.option("--n-rows", default=None, type=int)
def render_segmentation_yt(input_file, output_video, bin_size, fps, start_row, n_rows):
    """
    Renders YT video with random colors.
    """
    t, x, y, _ = load_data_as_arrays(input_file, 1.0, 'txt', start_row, n_rows)
    cluster_ids = load_cluster_ids_chunk(input_file, start_row, n_rows)

    local_sort = np.argsort(t)
    t, x, y, cluster_ids = t[local_sort], x[local_sort], y[local_sort], cluster_ids[local_sort]
    t_limits = (t[0], t[-1])

    x_sort_idx = np.argsort(x)
    t, x, y, cluster_ids = t[x_sort_idx], x[x_sort_idx], y[x_sort_idx], cluster_ids[x_sort_idx]

    # CALL UNIFIED YT (color_lookup=None)
    render_yt_discrete(t, x, y, cluster_ids, output_video, bin_size, fps, color_lookup=None, t_limits=t_limits)
    click.echo(f"Saved → {output_video}")


if __name__ == "__main__":
    cli()