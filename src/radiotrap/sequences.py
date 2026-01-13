import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import networkx as nx


def find_sequences(df, spatial_radius=20.0, max_time_gap=1e9, exclude_noise=True):
    """
    Finds chains of events of ANY length (A->B, A->B->C, etc).

    Parameters
    ----------
    df : pandas.DataFrame
        Classification dataframe with at least mean_x, mean_y, mean_t, Cluster_ID, class, total_energy.
    spatial_radius : float
        Maximum spatial distance (pixels) between successive events in a chain.
    max_time_gap : float
        Maximum allowed time difference between events in a chain, in nanoseconds.
    exclude_noise : bool
        If True, chains consisting only of Gamma-class events are removed.
    """
    print("Sorting data by time...")
    df = df.sort_values("mean_t").reset_index(drop=True)

    # Numpy arrays for speed
    coords = df[["mean_x", "mean_y"]].values
    times = df["mean_t"].values
    ids = df["Cluster_ID"].values

    # Graph to store links
    # Nodes = Indices in the dataframe
    G = nx.DiGraph()

    N = len(df)
    CHUNK_SIZE = 5000

    print(f"Scanning {N} events for links (R={spatial_radius}px, T={max_time_gap / 1e9:.2f}s)...")

    # --- STEP 1: FIND PAIRS (Sliding Window) ---
    for i in tqdm(range(0, N, CHUNK_SIZE), desc="Building Links"):
        idx_start = i
        idx_end = min(i + CHUNK_SIZE, N)

        t_limit = times[idx_end - 1] + max_time_gap
        idx_limit = np.searchsorted(times, t_limit, side='right')

        context_indices = np.arange(idx_start, idx_limit)
        if len(context_indices) < 2: continue

        context_coords = coords[context_indices]
        tree = cKDTree(context_coords)

        n_source = idx_end - idx_start
        source_coords = context_coords[:n_source]
        neighbors_list = tree.query_ball_point(source_coords, r=spatial_radius)

        for loc_idx, neighbors in enumerate(neighbors_list):
            if not neighbors: continue
            abs_idx_1 = idx_start + loc_idx

            for loc_neighbor in neighbors:
                abs_idx_2 = context_indices[loc_neighbor]

                if abs_idx_2 <= abs_idx_1: continue  # Future only

                dt = times[abs_idx_2] - times[abs_idx_1]
                if dt > max_time_gap: continue

                # Add Edge to Graph
                # We store the dataframe Index, not Cluster_ID, for easier retrieval later
                G.add_edge(abs_idx_1, abs_idx_2, dt=dt)

    # --- STEP 2: STITCH CHAINS ---
    print(f"Found {G.number_of_edges()} links. Stitching chains...")

    # Weakly connected components finds all isolated subgraphs (chains)
    # This automatically handles A->B->C or even branching A->B/A->C
    chains = list(nx.weakly_connected_components(G))

    print(f"Identified {len(chains)} unique chains.")

    # --- STEP 3: FORMAT OUTPUT ---
    results = []

    for chain_id, node_set in enumerate(tqdm(chains, desc="Exporting")):
        # Get all events in this chain
        indices = sorted(list(node_set))

        # Check if it's just noise (Pure Gamma chain)
        # We need to look up classes.
        chain_classes = df.loc[indices, "class"].values
        if exclude_noise:
            # If chain is ONLY Gammas, skip it
            if np.all(chain_classes == "Gamma"):
                continue

        # Add to results
        for step_num, idx in enumerate(indices):
            row = df.iloc[idx]

            # Calculate delta from previous step (if not first)
            dt_prev = 0
            dist_prev = 0
            if step_num > 0:
                prev_idx = indices[step_num - 1]
                dt_prev = (times[idx] - times[prev_idx]) / 1e9  # Seconds
                dist_prev = np.linalg.norm(coords[idx] - coords[prev_idx])

            results.append({
                "Chain_ID": chain_id,
                "Sequence_Index": step_num,  # 0=Start, 1=Second event...
                "Cluster_ID": row["Cluster_ID"],
                "Class": row["class"],
                "Energy": row["total_energy"],
                "Time_Abs": row["mean_t"],
                "Time_Delta": dt_prev,  # Time since previous event in chain
                "Dist_Delta": dist_prev,
                # Metadata for visualization
                "Thumbnail": row.get("thumbnail_png", ""),
                "Roundness": row.get("mask_roundness", 0),
                "Radius": row.get("max_radius", 0)
            })

    return pd.DataFrame(results)