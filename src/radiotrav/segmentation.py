import numpy as np
from tqdm import tqdm
import imageio.v3 as iio


# ============================================================
# HOT PIXEL DETECTION
# ============================================================

def detect_hot_pixels(x, y, percentile=99, multiplier=50):
    """
    Detect pixels with abnormally high event counts (dead/noisy pixels).

    A pixel is flagged when its event count exceeds
    percentile(active_counts, `percentile`) * `multiplier`.

    The reference point is the `percentile`-th percentile of counts across
    all pixels that fired at least once. A pixel must fire `multiplier` times
    more than that reference to be considered a hot pixel.

    Default (percentile=99, multiplier=50): flags pixels firing 50× more than
    the 99th-percentile pixel — conservative enough to skip borderline cases.

    Args:
        x, y:        Integer coordinate arrays (shape N).
        percentile:  Percentile of active-pixel counts used as the reference.
        multiplier:  How many times above the reference a pixel must be to be flagged.

    Returns:
        event_mask:  bool array (shape N), True for events at hot pixels.
        threshold:   float, the count threshold that was applied.
        hot_coords:  list of (x, y, count) tuples for each flagged pixel.
    """
    pixel_idx = y.astype(np.int32) * 256 + x.astype(np.int32)
    counts = np.bincount(pixel_idx, minlength=256 * 256)

    active = counts[counts > 0]
    threshold = np.percentile(active, percentile) * multiplier

    hot_flat = counts >= threshold
    event_mask = hot_flat[pixel_idx]

    hot_indices = np.flatnonzero(hot_flat)
    hot_coords = [
        (int(idx % 256), int(idx // 256), int(counts[idx]))
        for idx in hot_indices
    ]

    return event_mask, float(threshold), hot_coords


# ============================================================
# UNION-FIND UTILS
# ============================================================

def get_root(parents, i):
    # Path compression
    if parents[i] != i:
        parents[i] = get_root(parents, parents[i])
    return parents[i]


def union(parents, i, j):
    root_i = get_root(parents, i)
    root_j = get_root(parents, j)
    if root_i != root_j:
        parents[root_j] = root_i


def segment_events_spatiotemporal(t, x, y, time_window, spatial_radius=1):
    """
    Connects events that are neighbors in x,y (within spatial_radius)
    and neighbors in time (within time_window).
    """
    n = len(t)
    parents = np.arange(n, dtype=np.int32)

    # Grid to store the index of the *last* event seen at each pixel
    last_seen_idx = np.full((256, 256), -1, dtype=np.int32)

    # Generate search offsets based on radius
    # We use a square box for speed (Chebyshev distance)
    # radius=1 -> [-1, 0, 1] (3x3 box)
    # radius=2 -> [-2, -1, 0, 1, 2] (5x5 box)
    search_range = range(-spatial_radius, spatial_radius + 1)

    print(f"Segmenting {n} events (Time: {time_window}, Radius: {spatial_radius})...")

    for i in tqdm(range(n), desc="Linking"):
        cx, cy, ct = x[i], y[i], t[i]

        # Check neighbors within radius
        for dy in search_range:
            ny = cy + dy
            if ny < 0 or ny >= 256: continue

            for dx in search_range:
                nx = cx + dx
                if nx < 0 or nx >= 256: continue

                # Optimization: Skip self-check if radius is large,
                # but we need self-check (0,0) for temporal linking of the same pixel!

                # Who was here last?
                prev_idx = last_seen_idx[ny, nx]

                if prev_idx != -1:
                    # Check time distance
                    dt = ct - t[prev_idx]

                    if dt <= time_window:
                        union(parents, i, prev_idx)

        # Update lookup grid
        last_seen_idx[cy, cx] = i

    print("Resolving cluster IDs...")
    cluster_ids = np.zeros(n, dtype=np.int32)
    for i in range(n):
        cluster_ids[i] = get_root(parents, i)

    return cluster_ids
