import numpy as np
import pandas as pd


def detect_file_type(path):
    """
    Auto-detect file type from file extension.
    Returns 't3pa' for .t3pa/.t3p files, 'txt' otherwise.
    """
    lower = path.lower()
    if lower.endswith(".t3pa") or lower.endswith(".t3p"):
        return "t3pa"
    return "txt"


def load_data_as_arrays(path, ftoa_factor=1.0, file_type=None, start_row=0, n_rows=None):
    """
    Loads a specific chunk of rows from the file.
    
    Args:
        path: Path to the input file
        ftoa_factor: Fine time of arrival factor (not used for t3pa, kept for compatibility)
        file_type: 't3pa' or 'txt'. If None, auto-detected from file extension.
        start_row: Starting row index
        n_rows: Number of rows to load (None = all)
    """
    # Auto-detect file type if not provided
    if file_type is None:
        file_type = detect_file_type(path)
    
    print(f"Loading {n_rows if n_rows else 'all'} rows starting at {start_row} from {file_type} file...")

    # 1. Read just the header to get column names
    if file_type == 't3pa':
        # For t3pa files, read with tab separator first to check, then fall back to whitespace
        # Some t3pa files use tabs, some use spaces
        try:
            header_df = pd.read_csv(path, sep="\t", header=0, nrows=0, engine="python")
            sep_char = "\t"
        except:
            header_df = pd.read_csv(path, sep=r"\s+", header=0, nrows=0, engine="python")
            sep_char = r"\s+"
        col_names = list(header_df.columns)
        skip_header = 1
    else:
        # TXT files usually have headers too based on your previous examples
        # Use CSV reader to handle quoted fields correctly
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            header = next(reader)
            col_names = header
        skip_header = 1
        sep_char = None  # Will use CSV reader for txt files

    # 2. Read the specific chunk
    if file_type == 't3pa':
        # For t3pa files, use pandas with the detected separator
        df = pd.read_csv(
            path,
            sep=sep_char,
            header=None,  # We provide names manually
            names=col_names,
            skiprows=start_row + skip_header,
            nrows=n_rows,
            dtype=str,
            engine="python"
        )
    else:
        # For txt files with quoted fields, use CSV reader
        import csv
        rows = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            next(reader)  # Skip header
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
                rows.append(row)
                count += 1
        
        # Convert to DataFrame
        # Pad rows to same length if needed
        max_len = max(len(row) for row in rows) if rows else len(col_names)
        padded_rows = [row + [''] * (max_len - len(row)) for row in rows]
        df = pd.DataFrame(padded_rows, columns=col_names[:max_len] if len(col_names) >= max_len else col_names + [''] * (max_len - len(col_names)))
        df = df[col_names[:len(df.columns)]]  # Keep only the columns we need

    if len(df) == 0:
        raise ValueError("Loaded 0 rows. Check your --start-row index.")

    # 3. Clean and Parse
    if file_type == 't3pa':
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Handle "Matrix Index" being split into "Matrix" and "Index.1" (or similar)
        # Check if we have "Matrix Index" as a single column
        if "Matrix Index" in df.columns:
            matrix_col = "Matrix Index"
        elif "Matrix" in df.columns:
            # "Matrix Index" was split - use "Matrix" column
            matrix_col = "Matrix"
        else:
            raise ValueError(f"Could not find Matrix Index column. Available columns: {list(df.columns)}")
        
        x = (df[matrix_col] % 256).astype(np.int32).values
        y = (df[matrix_col] // 256).astype(np.int32).values
        # Match R script conversion: arrival_time = (ToA * 25) - ((FToA * 25)/16)
        # Factors originate from Advacam PIXET Wiki
        t = (df["ToA"] * 25.0 - (df["FToA"] * 25.0 / 16.0)).values

    elif file_type == 'txt':
        # Clean headers (remove quotes/dots)
        df.columns = df.columns.str.replace('"', '').str.replace("'", "").str.replace(".", "_").str.strip()
        df = df.rename(columns={"x_pos": "x", "y_pos": "y", "arrival_time": "time"})

        t = pd.to_numeric(df.get("time", 0), errors="coerce").fillna(0).values
        x = pd.to_numeric(df.get("x", 0), errors="coerce").fillna(0).astype(np.int32).values
        y = pd.to_numeric(df.get("y", 0), errors="coerce").fillna(0).astype(np.int32).values

    # Energy
    if "ToT" in df.columns:
        tot = pd.to_numeric(df["ToT"], errors="coerce").fillna(0).values.astype(np.float32)
    else:
        tot = np.ones_like(t, dtype=np.float32)

    # 4. Sort (Only strictly necessary if file isn't time-ordered,
    # but essential for the rest of the pipeline)
    sort_idx = np.argsort(t)

    return t[sort_idx], x[sort_idx], y[sort_idx], tot[sort_idx]