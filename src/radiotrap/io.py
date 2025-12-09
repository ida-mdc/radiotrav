import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


def load_data_as_arrays(path, ftoa_factor=1.0, file_type='t3pa', start_row=0, n_rows=None):
    """
    Loads a specific chunk of rows from the file.
    """
    print(f"Loading {n_rows if n_rows else 'all'} rows starting at {start_row}...")

    # 1. Read just the header to get column names
    if file_type == 't3pa':
        header_df = pd.read_csv(path, sep=r"\s+", header=0, nrows=0, engine="python")
        col_names = header_df.columns
        skip_header = 1
    else:
        # TXT files usually have headers too based on your previous examples
        header_df = pd.read_csv(path, sep=r"\s+", nrows=0, engine="python")
        col_names = header_df.columns
        skip_header = 1

    # 2. Read the specific chunk
    # We skip 'start_row' data lines + the header lines
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,  # We provide names manually
        names=col_names,
        skiprows=start_row + skip_header,
        nrows=n_rows,
        dtype=str,
        engine="python"
    )

    if len(df) == 0:
        raise ValueError("Loaded 0 rows. Check your --start-row index.")

    # 3. Clean and Parse
    if file_type == 't3pa':
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        matrix_col = "Matrix Index" if "Matrix Index" in df.columns else "Matrix"
        x = (df[matrix_col] % 256).astype(np.int32).values
        y = (df[matrix_col] // 256).astype(np.int32).values
        t = (df["ToA"] + df["FToA"] * ftoa_factor).values

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