# RadioTRAV: Time Resolved Autoradiography Visualization for Nuclear Radiation

RadioTRAV (Time Resolved Autoradiography Visualization) processes, analyzes, and visualizes nuclear radiation data from **Timepix3** detectors (e.g. `.t3pa` or derived `.txt` / segmented text files).

- **Rendering**: raw event heatmaps to MP4 or TIFF.
- **Segmentation**: cluster ID assignment for individual tracks.
- **Classification**: Alpha / Beta / Gamma / Other, with per-cluster metrics.
- **Sequence analysis**: decay-chain style sequences in space and time.
- **Reporting**: HTML dashboard for classification and sequences.

---

## Installation

RadioTRAV uses **uv** for environment and dependency management.

### Install `uv`

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing so `uv` is on your `PATH`.

### Get the code

Clone this repository and `cd` into the project folder:

```bash
git clone https://github.com/ida-mdc/radiotrav.git
cd radiotrav
```

### Install dependencies

From the project folder:

```bash
uv sync
uv pip install -e .
```

The `radiotrav` command-line tool is now available as:

```bash
uv run radiotrav ...
```

---

## Quickstart

### 1. Processing

Run the complete pipeline (segmentation, classification, and sequence analysis):

```bash
uv run radiotrav process input_events.t3pa output_dir --time-window 100 --spatial-radius 2
```

This creates `segmented.txt`, `classification.csv`, and `chains.csv` in the output directory.

**Customizing classification parameters:**

You can adjust the thresholds used for Alpha/Beta/Gamma classification:

```bash
uv run radiotrav process input_events.t3pa output_dir --time-window 100 \
  --gamma-max-area 10 \
  --alpha-min-radius 1.5 \
  --alpha-min-roundness 0.85 \
  --beta-max-radius 5.0 \
  --beta-min-dimension 6
```

See the [Command Reference](#command-reference) below for all options.

### 2. Dashboard

Open `docs/dashboard/index.html` in your web browser. Load the three generated files from your output directory:
- **segmented.txt** - Events with cluster assignments
- **classification.csv** - Classification results
- **chains.csv** - Sequence/chain analysis results

The dashboard provides classification plots, galleries, sequence analysis, and an interactive event viewer.

### 3. Video Rendering

Render videos with filtering options:

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

See the [Command Reference](#command-reference) below for all render options.

---

## Comprehensive Documentation

### Updating

After pulling code updates or making changes to the source code:

```bash
git pull
uv sync
uv pip install -e .
```

### Command Reference {#command-reference}

All commands are invoked as `uv run radiotrav <command> ...`.

#### `process`

Run the complete pipeline: segmentation, classification, and sequence analysis.

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
  - `--skip-existing-segmentation`: skip segmentation if `segmented.txt` already exists in output directory.
  - `--skip-existing-classification`: skip classification if `classification.csv` already exists in output directory.
  - **Classification parameters:**
    - `--gamma-max-area [FLOAT]` (default: `9.0`): maximum mask_area for Gamma classification. Clusters with area ≤ this value are classified as Gamma.
    - `--alpha-min-radius [FLOAT]` (default: `1.0`): minimum max_radius for Alpha classification. Alpha clusters must have max_radius > this value.
    - `--alpha-min-roundness [FLOAT]` (default: `0.9`): minimum mask_roundness for Alpha classification. Alpha clusters must have roundness > this value.
    - `--beta-max-radius [FLOAT]` (default: `4.0`): maximum max_radius for Beta classification. Beta clusters must have max_radius < this value.
    - `--beta-min-dimension [FLOAT]` (default: `5.0`): minimum width or height for Beta classification. Beta clusters must have width > this OR height > this.

#### `render`

Render events to video/image with filtering options matching the dashboard.

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

### Working with Large Files

Both `process` and `render` support partial loading of very large input files using `--start-row` and `--n-rows`:

Example: render only the first 10,000 events to check settings:

```bash
uv run radiotrav render input.txt test.mp4 --n-rows 10000
```

Use the same pattern with `process` when testing on subsets of large datasets.
