import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import networkx as nx
from pathlib import Path


def load_pattern_lookup(pattern_file):
    """
    Load pattern lookup file (CSV format).
    
    Expected format:
        pattern_name,pattern_sequence
        "Alpha Decay","Alpha,Alpha"
        "Beta Decay Chain","Alpha,Beta,Gamma"
    
    Or with arrow notation:
        pattern_name,pattern_sequence
        "Alpha Decay","Alpha->Alpha"
        "Beta Decay Chain","Alpha->Beta->Gamma"
    
    Parameters
    ----------
    pattern_file : str or Path
        Path to CSV file with pattern definitions.
    
    Returns
    -------
    list of dict
        List of dictionaries with 'name' and 'pattern' (list of class names) keys.
    """
    if pattern_file is None or not Path(pattern_file).exists():
        return []
    
    try:
        df_patterns = pd.read_csv(pattern_file)
        
        # Check required columns
        if 'pattern_name' not in df_patterns.columns or 'pattern_sequence' not in df_patterns.columns:
            print(f"Warning: Pattern file must have 'pattern_name' and 'pattern_sequence' columns. Skipping pattern matching.")
            return []
        
        patterns = []
        for _, row in df_patterns.iterrows():
            pattern_str = str(row['pattern_sequence']).strip()
            # Support both comma and arrow notation
            if '->' in pattern_str:
                pattern = [p.strip() for p in pattern_str.split('->')]
            else:
                pattern = [p.strip() for p in pattern_str.split(',')]
            
            patterns.append({
                'name': str(row['pattern_name']).strip(),
                'pattern': pattern
            })
        
        print(f"Loaded {len(patterns)} patterns from {pattern_file}")
        return patterns
    except Exception as e:
        print(f"Warning: Could not load pattern file {pattern_file}: {e}. Skipping pattern matching.")
        return []


def match_chain_pattern(chain_classes, patterns):
    """
    Match a chain's class sequence against known patterns.
    
    Parameters
    ----------
    chain_classes : list or array
        List of class names in the chain (e.g., ['Alpha', 'Beta', 'Gamma']).
    patterns : list of dict
        List of pattern dictionaries with 'name' and 'pattern' keys.
    
    Returns
    -------
    str or None
        Name of the first matching pattern, or None if no match.
    """
    chain_list = list(chain_classes)
    
    for pat in patterns:
        pat_seq = pat['pattern']
        # Exact match
        if chain_list == pat_seq:
            return pat['name']
        # Check if pattern is a subsequence (allowing extra events)
        # This allows matching "Alpha->Beta" in "Alpha->Beta->Gamma"
        if len(pat_seq) <= len(chain_list):
            # Check if pattern matches the beginning
            if chain_list[:len(pat_seq)] == pat_seq:
                return pat['name']
    
    return None


def find_sequences(df, spatial_radius=20.0, max_time_gap=1e9, exclude_noise=True, max_chain_length=None, pattern_lookup_file=None):
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
    max_chain_length : int, optional
        Maximum number of events allowed in a chain. Chains longer than this will be ignored.
        If None, no limit is applied.
    pattern_lookup_file : str or Path, optional
        Path to CSV file containing pattern definitions for matching specific chemical reactions.
        Expected format: pattern_name,pattern_sequence (e.g., "Alpha Decay","Alpha,Alpha")
    """
    print("Sorting data by time...")
    df = df.sort_values("mean_t").reset_index(drop=True)

    # Load pattern lookup if provided
    patterns = load_pattern_lookup(pattern_lookup_file)

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
    skipped_long = 0
    skipped_noise = 0

    for chain_id, node_set in enumerate(tqdm(chains, desc="Exporting")):
        # Get all events in this chain
        indices = sorted(list(node_set))
        chain_length = len(indices)

        # Check maximum chain length
        if max_chain_length is not None and chain_length > max_chain_length:
            skipped_long += 1
            continue

        # Check if it's just noise (Pure Gamma chain)
        # We need to look up classes.
        chain_classes = df.loc[indices, "class"].values
        if exclude_noise:
            # If chain is ONLY Gammas, skip it
            if np.all(chain_classes == "Gamma"):
                skipped_noise += 1
                continue

        # Match against patterns if provided
        pattern_name = None
        if patterns:
            pattern_name = match_chain_pattern(chain_classes, patterns)

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

            result_row = {
                "Chain_ID": chain_id,
                "Sequence_Index": step_num,  # 0=Start, 1=Second event...
                "Cluster_ID": row["Cluster_ID"],
                "Class": row["class"],
                "Energy": row["total_energy"],
                "Time_Abs": row["mean_t"],
                "Time_Delta": dt_prev,  # Time since previous event in chain
                "Dist_Delta": dist_prev,  # Spatial distance from previous event in chain (pixels)
                # Metadata for visualization
                "Thumbnail": row.get("thumbnail_png", ""),
                "Roundness": row.get("mask_roundness", 0),
                "Radius": row.get("max_radius", 0)
            }
            
            # Add pattern name if matched
            if pattern_name:
                result_row["Pattern_Name"] = pattern_name
            else:
                result_row["Pattern_Name"] = ""
            
            results.append(result_row)

    if skipped_long > 0:
        print(f"Skipped {skipped_long} chains exceeding maximum length ({max_chain_length}).")
    if skipped_noise > 0:
        print(f"Skipped {skipped_noise} pure Gamma chains (noise).")

    return pd.DataFrame(results)