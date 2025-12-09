import numpy as np
import pandas as pd
import base64
import io
from tqdm import tqdm
import matplotlib.cm as cm
from PIL import Image


# ============================================================
# CLUSTERING & CLASSIFICATION
# ============================================================

def generate_thumbnail_base64(group):
    """
    Generates a Base64 encoded PNG thumbnail (heatmap) for a single cluster.
    """
    # 1. Determine bounding box
    min_x, max_x = group["x_pos"].min(), group["x_pos"].max()
    min_y, max_y = group["y_pos"].min(), group["y_pos"].max()

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 2. Create grid (Max Projection)
    # We use float32 for accumulation
    grid = np.zeros((height, width), dtype=np.float32)

    # Local coordinates
    local_x = group["x_pos"].values - min_x
    local_y = group["y_pos"].values - min_y
    energy = group["ToT"].values

    # 3. Fill Grid (Vectorized accumulation is tricky without looping or specialized lib,
    # but for small thumbnails, simple iteration or np.add.at is fine)
    # np.add.at is fast unbuffered summation
    np.add.at(grid, (local_y, local_x), energy)

    # 4. Colorize (Turbo Colormap)
    # Normalize 0..1
    max_val = grid.max()
    if max_val > 0:
        grid /= max_val

    cmap = cm.get_cmap("turbo")
    rgba = cmap(grid)  # Returns (H, W, 4) floats 0..1

    # 5. Convert to Image
    # Flatten alpha or just use RGB. Turbo doesn't have transparency, but background is 0 energy.
    # Let's make 0 energy black.
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    # Mask background (where energy was 0)
    mask = grid == 0
    rgb[mask] = 0

    img = Image.fromarray(rgb)

    # 6. Encode to Base64 PNG
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def classify_clusters(df):
    """
    Classifies clusters and calculates detailed metrics + thumbnails.
    """
    print("Calculating cluster metrics...")

    # --- 1. Basic Stats (Vectorized) ---
    # We calculate bounding box and centroids first
    basic_stats = df.groupby("Cluster_ID").agg({
        "x_pos": ["min", "max", "mean", "count"],
        "y_pos": ["min", "max", "mean"],
        "arrival_time": "mean",  # Mean Time
        "ToT": ["sum", "mean"]  # Total Energy, Density
    })

    # Flatten columns
    basic_stats.columns = [
        "min_x", "max_x", "mean_x", "area",
        "min_y", "max_y", "mean_y",
        "mean_t",
        "total_energy", "mean_energy"
    ]

    # Derived Dimensions
    basic_stats["width"] = basic_stats["max_x"] - basic_stats["min_x"] + 1
    basic_stats["height"] = basic_stats["max_y"] - basic_stats["min_y"] + 1

    # --- 2. Shape Analysis (Eigenvalues / Covariance) ---
    # We compute covariance matrix terms manually on the original DF
    # Map means back to rows for calculation
    df["mx"] = df["Cluster_ID"].map(basic_stats["mean_x"])
    df["my"] = df["Cluster_ID"].map(basic_stats["mean_y"])

    df["dx"] = df["x_pos"] - df["mx"]
    df["dy"] = df["y_pos"] - df["my"]

    df["cov_xx"] = df["dx"] * df["dx"]
    df["cov_yy"] = df["dy"] * df["dy"]
    df["cov_xy"] = df["dx"] * df["dy"]

    cov_stats = df.groupby("Cluster_ID").agg({
        "cov_xx": "mean",
        "cov_yy": "mean",
        "cov_xy": "mean"
    })

    # Merge
    metrics = pd.concat([basic_stats, cov_stats], axis=1)

    # Eigenvalues
    t = metrics["cov_xx"] + metrics["cov_yy"]
    d = metrics["cov_xx"] * metrics["cov_yy"] - metrics["cov_xy"] ** 2
    gap = np.sqrt(np.maximum(0, t * t - 4 * d))

    l1 = (t + gap) / 2
    l2 = (t - gap) / 2
    l1 = l1.replace(0, 1e-9)

    # Aspect Ratio (Roundness of max projection)
    metrics["aspect_ratio"] = np.sqrt(l2 / l1)

    # --- 3. Thumbnail Generation ---
    # This cannot be easily vectorized, so we apply per group.
    # To speed it up, we only do it if the dataframe isn't massive,
    # or show a progress bar.
    print("Generating thumbnails...")
    tqdm.pandas(desc="Thumbnails")
    metrics["thumbnail_png"] = df.groupby("Cluster_ID").progress_apply(generate_thumbnail_base64)

    # --- 4. Classification Rules ---
    def get_class(row):
        area = row["area"]
        ratio = row["aspect_ratio"]
        # density = row["mean_energy"]

        if area <= 4: return "Gamma"
        if ratio > 0.6: return "Alpha"
        return "Beta"

    metrics["class"] = metrics.apply(get_class, axis=1)

    # Cleanup: return relevant columns
    return metrics[[
        "class",
        "mean_t",
        "min_x", "max_x", "min_y", "max_y", "width", "height", "mean_x", "mean_y",
        "aspect_ratio", "area", "total_energy", "mean_energy",
        "thumbnail_png"
    ]]

