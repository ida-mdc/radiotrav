import numpy as np
import pandas as pd
import base64
import io
from tqdm import tqdm
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import distance_transform_edt


# ============================================================
# ADVANCED CLUSTER ANALYSIS
# ============================================================

def analyze_cluster_shape(group):
    """
    Generates Thumbnail + Calculates Advanced Morphological Metrics.
    Returns a Series to be merged into the main stats.
    """
    # 1. Determine bounding box
    min_x, max_x = group["x_pos"].min(), group["x_pos"].max()
    min_y, max_y = group["y_pos"].min(), group["y_pos"].max()

    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    # 2. Create Energy Grid (Float32 for accumulation)
    grid = np.zeros((height, width), dtype=np.float32)

    local_x = (group["x_pos"].values - min_x).astype(int)
    local_y = (group["y_pos"].values - min_y).astype(int)
    energy = group["ToT"].values.astype(np.float32)

    # Fill grid with energy
    np.add.at(grid, (local_y, local_x), energy)

    # 3. Create Binary Mask (0/1) for Shape Analysis
    binary_mask = (grid > 0).astype(np.uint8)

    # --- METRIC: Max Thickness (Distance Transform) ---
    # Calculates distance from every '1' pixel to the nearest '0'.
    # The max value is the radius of the largest inscribed circle.
    if binary_mask.sum() > 0:
        dist_map = distance_transform_edt(binary_mask)
        max_radius = dist_map.max()
    else:
        max_radius = 0.0

    # --- METRIC: Mask Roundness (Eigenvalues of the shape itself) ---
    # We calculate the shape tensor of the mask (unweighted by energy)
    # This handles "is the footprint round?" better than energy-weighted moments.
    y_idxs, x_idxs = np.nonzero(binary_mask)
    if len(x_idxs) > 2:
        # Center the coordinates
        xc = x_idxs - x_idxs.mean()
        yc = y_idxs - y_idxs.mean()

        # Covariance matrix terms
        c_xx = (xc * xc).mean()
        c_yy = (yc * yc).mean()
        c_xy = (xc * yc).mean()

        # Eigenvalues
        tr = c_xx + c_yy
        det = c_xx * c_yy - c_xy ** 2
        gap = np.sqrt(max(0, tr * tr - 4 * det))
        l1 = (tr + gap) / 2
        l2 = (tr - gap) / 2

        mask_roundness = np.sqrt(l2 / l1) if l1 > 0 else 0
    else:
        mask_roundness = 1.0  # Single point is round

    # --- THUMBNAIL GENERATION ---
    # Normalize Energy Grid for Visualization
    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    cmap = cm.get_cmap("turbo")
    rgba = cmap(grid)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    # Apply mask to make background black
    rgb[binary_mask == 0] = 0

    # Encode
    img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    return pd.Series({
        "thumbnail_png": b64_str,
        "max_radius": max_radius,  # Thickness (Radius)
        "mask_roundness": mask_roundness,  # Geometric Roundness
        "mask_area": binary_mask.sum()  # Exact pixel count
    })


def classify_clusters(df):
    """
    Classifies clusters using advanced topological metrics.
    """
    print("Calculating basic vector stats...")

    # --- 1. Fast Vectorized Stats (Centroids & Bounding Box) ---
    basic_stats = df.groupby("Cluster_ID").agg({
        "x_pos": ["min", "max", "mean"],
        "y_pos": ["min", "max", "mean"],
        "arrival_time": "mean",
        "ToT": ["sum", "mean"]
    })

    basic_stats.columns = [
        "min_x", "max_x", "mean_x",
        "min_y", "max_y", "mean_y",
        "mean_t",
        "total_energy", "mean_energy"
    ]

    # Dimensions
    basic_stats["width"] = basic_stats["max_x"] - basic_stats["min_x"] + 1
    basic_stats["height"] = basic_stats["max_y"] - basic_stats["min_y"] + 1

    # --- 2. Advanced Shape Analysis (Slow Loop) ---
    # We apply the detailed analysis per cluster
    print("Analyzing shapes (Thickness, Thumbnails)...")
    tqdm.pandas(desc="Shape Analysis")

    shape_metrics = df.groupby("Cluster_ID").progress_apply(analyze_cluster_shape)

    # Join the fast stats with the slow shape metrics
    metrics = pd.concat([basic_stats, shape_metrics], axis=1)

    # --- 3. Physics-Based Classification ---
    def get_class(row):
        # 1. GAMMA: Tiny spots (Low Area)
        # Note: We use mask_area (unique pixels)
        if row["mask_area"] <= 9:
            return "Gamma"

        # 2. ALPHA: Thick and Round
        # Thickness Check: Radius must be significant > 1.5 pixels (Diameter > 3px)
        # Roundness Check: Must be geometrically round > 0.6
        if row["max_radius"] > 1 and row["mask_roundness"] > 0.6:
            return "Alpha"

        # 3. BETA: Thin tracks
        # Even if they curl up and look "round" in aspect ratio,
        # their thickness (max_radius) will remain low (approx 1.0 - 1.5).
        if row["max_radius"] < 2 and (row["width"] > 5 or row["height"] > 5):
            return "Beta"

        return "Other"

    print("Classifying...")
    metrics["class"] = metrics.apply(get_class, axis=1)

    # Return sorted by time
    return metrics[[
        "class",
        "mean_t",
        "total_energy", "mean_energy",
        "mask_area", "mask_roundness", "max_radius",
        "min_x", "max_x", "min_y", "max_y", "width", "height", "mean_x", "mean_y",
        "thumbnail_png"
    ]].sort_values("mean_t")