# RadioTrap: Radiation Event Processor

RadioTrap processes, analyzes, and visualizes radiation data from **Timepix3** detectors (e.g. `.t3pa` or derived `.txt` / segmented text files).

- **Rendering**: raw event heatmaps to MP4 or TIFF.
- **Segmentation**: cluster ID assignment for individual tracks.
- **Classification**: Alpha / Beta / Gamma / Other, with per-cluster metrics.
- **Sequence analysis**: decay-chain style sequences in space and time.
- **Reporting**: HTML dashboard for classification and sequences.

---

## Installation

RadioTrap uses **uv** for environment and dependency management.

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
git clone https://gitlab.com/ida-mdc/radiotrap.git
cd radiotrap
```

### Install dependencies

From the project folder:

```bash
uv sync
uv pip install -e .
```

The `radiotrap` command-line tool is now available as:

```bash
uv run radiotrap ...
```

---

## Quickstart

### 1. Processing

Run the complete pipeline (segmentation, classification, and sequence analysis):

```bash
uv run radiotrap process input_events.t3pa output_dir --time-window 100 --spatial-radius 2
```

This creates `segmented.txt`, `classification.csv`, `chains.csv`, and `dashboard.html` in the output directory.

See the [Command Reference](#command-reference) below for all options.

### 2. Dashboard

Open `report/dashboard.html` in your web browser (this file is also copied to the output directory for portability). Load the three generated files from your output directory:
- **segmented.txt** - Events with cluster assignments
- **classification.csv** - Classification results
- **chains.csv** - Sequence/chain analysis results

The dashboard provides classification plots, galleries, sequence analysis, and an interactive event viewer.

### 3. Video Rendering

Render videos with filtering options:

```bash
# Basic rendering
uv run radiotrap render output_dir output.mp4

# Filter by radiation type
uv run radiotrap render output_dir alpha_only.mp4 --radiation Alpha

# Filter by sequence pattern
uv run radiotrap render output_dir beta_alpha_chains.mp4 \
  --seq-query "Beta - Alpha" --seq-exact

# Custom projection and mode
uv run radiotrap render output_dir energy_sideview.mp4 \
  --projection yt --mode energy --bin-size 1000000
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

All commands are invoked as `uv run radiotrap <command> ...`.

#### `process`

Run the complete pipeline: segmentation, classification, and sequence analysis.

- **Usage:** `uv run radiotrap process INPUT_FILE OUTPUT_DIR --time-window ...`
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

#### `render`

Render events to video/image with filtering options matching the dashboard.

- **Usage:** `uv run radiotrap render INPUT_PATH OUTPUT_FILE [options]`
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
  - `--mode [classification|energy|density|time]` (default: `classification`): display mode.
  - `--bin-size [FLOAT]`: time bin size in nanoseconds (e.g., `10000000` for 10ms). If not set, derived from `--speed` and `--fps`.
  - `--time-window [FLOAT]`: fade in/out window in nanoseconds (e.g., `1e9` for 1s). Events within this window around each frame center are blended with alpha fade.
  - `--speed [FLOAT]` (default: `1.0`): playback speed multiplier. Higher = fewer frames, faster render. Only used when `--bin-size` is not set.
  - `--fps [INT]` (default: `30`): frames per second (for MP4 format).
  - `--start-row [INT]` (default: `0`): first input row to load.
  - `--n-rows [INT]` (default: all): number of rows to process.

### Working with Large Files

Both `process` and `render` support partial loading of very large input files using `--start-row` and `--n-rows`:

Example: render only the first 10,000 events to check settings:

```bash
uv run radiotrap render input.txt test.mp4 --n-rows 10000
```

Use the same pattern with `process` when testing on subsets of large datasets.
