# RadioTRAV: Time Resolved Autoradiography Visualization for Nuclear Radiation

RadioTRAV (Time Resolved Autoradiography Visualization) processes, analyzes, and visualizes nuclear radiation data from **Timepix3** detectors (e.g. `.t3pa` or derived `.txt` / segmented text files).

- **Segmentation**: cluster ID assignment for individual tracks.
- **Classification**: Alpha / Beta / Gamma / Other, with per-cluster metrics.
- **Sequence analysis**: decay-chain style sequences in space and time.
- **Dashboard**: interactive web viewer for classification and sequences.
- **Rendering**: export to MP4 or PNG.

---

## Installation

RadioTRAV uses **uv** for environment and dependency management.

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing so `uv` is on your `PATH`.

Clone this repository and install:

```bash
git clone https://github.com/ida-mdc/radiotrav.git
cd radiotrav
uv sync
uv pip install -e .
```

The `radiotrav` command-line tool is now available as `uv run radiotrav ...`.

To update after pulling new code:

```bash
git pull
uv sync
uv pip install -e .
```

---

## Processing

### Running the pipeline

Run the complete pipeline (segmentation, classification, and sequence analysis):

```bash
uv run radiotrav process input_events.t3pa output_dir --time-window 100 --spatial-radius 2
```

This creates `segmented.txt`, `classification.csv`, `chains.csv`, and `hot_pixels.csv` in the output directory.

You can adjust the thresholds used for Alpha/Beta/Gamma classification:

```bash
uv run radiotrav process input_events.t3pa output_dir --time-window 100 \
  --gamma-max-area 10 \
  --alpha-min-radius 1.5 \
  --alpha-min-roundness 0.85 \
  --beta-max-radius 5.0 \
  --beta-min-dimension 6
```

### Pipeline steps

#### 1. Hot pixel masking

Before segmentation, pixels that fire far more often than the rest of the detector are identified as dead or noisy and removed. The threshold is **50 × the 99th percentile** of per-pixel event counts across the whole dataset: a pixel must fire at least 50× more than a normally-busy pixel to be masked. The masked pixels and their event counts are written to `hot_pixels.csv` in the output directory. This step can be disabled with `--no-hot-pixel-filter`.

#### 2. Spatiotemporal segmentation

Events are grouped into clusters by linking each event to any neighbour that fired within `--spatial-radius` pixels and `--time-window` nanoseconds of it. The result is written to `segmented.txt`.

#### 3. Cluster classification

Each cluster is described by its shape (bounding box, pixel area, maximum inscribed-circle radius, geometric roundness) and classified as one of:

- **Gamma** — small spot (area ≤ `--gamma-max-area`)
- **Alpha** — large, round, thick deposit (`max_radius > --alpha-min-radius` and `roundness > --alpha-min-roundness`)
- **Beta** — elongated thin track (`max_radius < --beta-max-radius` and at least one dimension > `--beta-min-dimension`)
- **Other** — anything that does not fit the above

Results are written to `classification.csv`.

#### 4. Sequence analysis

Clusters that are spatially and temporally close to each other are linked into chains (decay sequences). Chains are written to `chains.csv`.

### Command reference: `process`

- **Usage:** `uv run radiotrav process INPUT_FILE OUTPUT_DIR --time-window ...`
- **Options:**
  - `--time-window [FLOAT]` (required): maximum time gap **in nanoseconds** between pixels in the same cluster.
  - `--spatial-radius [INT]` (default: `1`): spatial search radius in pixels.
  - `--sequence-radius [FLOAT]` (default: `15.0`): spatial radius for sequence analysis in pixels.
  - `--sequence-time-window [FLOAT]` (default: `1e9`): maximum time gap between events in a chain in nanoseconds.
  - `--sequence-max-length [INT]`: maximum number of events in a chain (longer chains ignored).
  - `--sequence-pattern-lookup [PATH]`: CSV file with pattern definitions for sequence matching.
  - `--start-row [INT]` (default: `0`): first input row to load.
  - `--n-rows [INT]` (default: all): number of rows to process.
  - `--no-hot-pixel-filter`: skip the hot pixel masking step (see above).
  - `--skip-existing-segmentation`: skip segmentation if `segmented.txt` already exists in output directory.
  - `--skip-existing-classification`: skip classification if `classification.csv` already exists in output directory.
  - **Classification parameters:**
    - `--gamma-max-area [FLOAT]` (default: `9.0`): maximum mask_area for Gamma classification. Clusters with area ≤ this value are classified as Gamma.
    - `--alpha-min-radius [FLOAT]` (default: `3.0`): minimum max_radius for Alpha classification. Alpha clusters must have max_radius > this value.
    - `--alpha-min-roundness [FLOAT]` (default: `0.85`): minimum mask_roundness for Alpha classification. Alpha clusters must have roundness > this value.
    - `--beta-max-radius [FLOAT]` (default: `4.0`): maximum max_radius for Beta classification. Beta clusters must have max_radius < this value.
    - `--beta-min-dimension [FLOAT]` (default: `5.0`): minimum width or height for Beta classification. Beta clusters must have width > this OR height > this.

### Video rendering

Export results to MP4 or PNG with the same filtering options as the dashboard:

```bash
# Basic rendering
uv run radiotrav render output_dir output.mp4

# Filter by radiation type
uv run radiotrav render output_dir alpha_only.mp4 --radiation Alpha

# Filter by sequence pattern
uv run radiotrav render output_dir beta_alpha_chains.mp4 \
  --seq-query "Beta - Alpha" --seq-exact

# Custom projection and mode
uv run radiotrav render output_dir energy_sideview.mp4 \
  --projection yt --mode energy --speed 100
```

### Command reference: `render`

- **Usage:** `uv run radiotrav render INPUT_PATH OUTPUT_FILE [options]`
- **Input:** `INPUT_PATH` can be a project output directory (with `segmented.txt`, `classification.csv`, `chains.csv`) or a single events file.
- **Options:**
  - `--classification-csv [PATH]`: classification CSV (optional if input is project output dir).
  - `--chains-csv [PATH]`: chains CSV for sequence filter (optional if input is project output dir).
  - `--radiation [ALL|Alpha|Beta|Gamma|Other]` (default: `ALL`): filter by radiation type.
  - `--seq-query [STRING]`: filter chains by signature (e.g., `'Alpha - Beta'`).
  - `--seq-exact`: match chain signature exactly (otherwise contains).
  - `--seq-max-dt [FLOAT]`: max time delta in chain (seconds).
  - `--seq-max-dist [FLOAT]`: max spatial distance in chain (pixels).
  - `--view [animated|max]` (default: `animated`): `animated` (MP4) or `max` (PNG projection).
  - `--projection [xy|yt]` (default: `xy`): `xy` (top-down) or `yt` (side view, Y vs Time).
  - `--mode [classification|energy]` (default: `classification`): display mode.
  - `--time-window [FLOAT]`: fade in/out window in nanoseconds (e.g., `1e9` for 1s). Events within this window around each frame center are blended with alpha fade.
  - `--speed [FLOAT]` (default: `1.0`): playback speed multiplier. Higher = fewer frames, faster render.
  - `--fps [INT]` (default: `30`): frames per second (for MP4 format).
  - `--start-row [INT]` (default: `0`): first input row to load.
  - `--n-rows [INT]` (default: all): number of rows to process.

### Working with large files

Both `process` and `render` support partial loading of very large input files using `--start-row` and `--n-rows`:

```bash
uv run radiotrav render input.txt test.mp4 --n-rows 10000
```

---

## Dashboard

The dashboard is a single-page web app. You can open it in two ways:

- **Hosted version (no install needed):** [https://ida-mdc.github.io/radiotrav/dashboard/](https://ida-mdc.github.io/radiotrav/dashboard/)
- **Local version:** open `docs/dashboard/index.html` from this repository in your browser

All file loading happens client-side — no data is uploaded anywhere.

### Loading files

Use the three file pickers in the top-right header to load the outputs from `radiotrav process`:

| Field | File | Required for |
|---|---|---|
| segmented.txt (events) | `segmented.txt` | Event viewer |
| classification.csv | `classification.csv` | Classification tabs, viewer coloring |
| chains.csv | `chains.csv` | Sequence tabs |

Files can be loaded in any order and independently.

### Global filters

The filter bar below the header applies to all tabs and the event viewer simultaneously:

- **Classification** — show only one radiation type (Alpha / Beta / Gamma / Other) or all
- **Sequence pattern filter** — text filter on chain signatures, e.g. `Beta - Alpha`; toggle *full match* to require an exact match instead of a substring match
- **Max Δt (s)** — exclude chains where any step exceeds this time gap
- **Max Δd (px)** — exclude chains where any step exceeds this spatial distance
- **Reset filters** — clears all filters and selections

The status bar on the left of the filter row shows `events: N | clusters: N | chains: N`, updating as filters change.

### Left panel tabs

**Classification Plots** — aggregated statistics for the loaded classification data:
- Summary counts (total clusters, counts per radiation type, mean energy)
- Bar chart: number of clusters per class
- Histogram: energy (ToT) distribution, grouped by radiation type
- Histogram: max radius distribution, grouped by radiation type

**Classification Gallery** — thumbnail grid of individual clusters:
- Sort by any numeric column (energy, radius, etc.), ascending or descending
- Switch between *Grid* (image + label) and *Compact* (image-only mosaic) layouts
- Click any cluster to jump the event viewer to that cluster's position in time

**Sequence Plots** — aggregated statistics for decay chains:
- Bar chart: most frequent chain patterns (e.g. `Alpha - Beta`)
- Histogram: chain lengths
- Histogram: time gaps between events within chains

**Sequence Gallery** — when a sequence filter is active, shows individual matching chains; otherwise shows aggregated patterns grouped by signature:
- Click a pattern or chain to select it; the event viewer and classification gallery update to show only those clusters

### Event viewer (right panel)

The event viewer shows the detector as a 256×256 pixel canvas, animated over time.

- **View mode:**
  - *Animated (fade)* — events fade in and out around the current time center
  - *Animated (sum)* — events accumulate within the time window
  - *Max projection* / *Sum projection* — static image over the full dataset
- **Display mode:**
  - *Classification* — pixels colored by cluster type (red = Alpha, cyan = Beta, green = Gamma, grey = Other)
  - *Energy (ToT)* — pixels colored by Time-over-Threshold value
  - *Density* — pixels colored by event count
  - *Time* — pixels colored by arrival time
- **Colormap** — selectable for Energy, Density, and Time modes (Turbo, Viridis, Plasma, etc.)
- **Time center / Window size** — scrub through time and control how wide a time slice is shown
- **Speed** — controls playback rate; drag the slider or press Play
- **Fullscreen** — the ⛶ button in the top-right corner of the canvas

The **Processing Metadata** card below the viewer shows the segmentation and classification parameters that were embedded in the CSV files, so you always know how the data was produced.
