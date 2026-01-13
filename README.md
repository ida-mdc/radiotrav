# RadioTrap: Radiation Event Processor

RadioTrap processes, analyzes, and visualizes radiation data from **Timepix3** detectors (e.g. `.t3pa` or derived `.txt` / segmented text files).

- **Rendering**: raw event heatmaps to MP4 or TIFF.
- **Segmentation**: cluster ID assignment for individual tracks.
- **Classification**: Alpha / Beta / Gamma / Other, with per-cluster metrics.
- **Sequence analysis**: decay-chain style sequences in space and time.
- **Reporting**: HTML reports for classification and sequences.

---

### 1. Installation

RadioTrap uses **uv** for environment and dependency management.

#### 1.1 Install `uv`

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing so `uv` is on your `PATH`.

#### 1.2 Get the code

Clone this repository and `cd` into the project folder:

```bash
git clone https://gitlab.com/ida-mdc/radiotrap.git
cd radiotrap
```

#### 1.3 Install dependencies

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

### 2. Updating

After pulling code updates or making changes to the source code, update the installed package:

```bash
git pull
uv sync
uv pip install -e .
```

---

### 3. Quick start

All commands below are run as `uv run radiotrap <command> ...`.

#### Phase A: segmentation and classification

1. **Segment** raw events into clusters:

   - `--time-window [FLOAT]`: maximum time gap **in nanoseconds** between pixels in the same cluster.
   - `--spatial-radius [INT]`: search radius in pixels (typically `1` or `2`).

   ```bash
   uv run radiotrap segment input_events.txt segmented.txt --time-window 100 --spatial-radius 2
   ```

2. **Classify** clusters (Alpha / Beta / Gamma / Other) and compute per-cluster metrics:

   ```bash
   uv run radiotrap classify segmented.txt classification.csv
   ```

3. **Find sequences** (decay-like chains) in the classified clusters:

   - `--radius [FLOAT]`: maximum spatial distance in pixels between successive clusters in a chain.
   - `--time-window [FLOAT]`: maximum time difference in **nanoseconds** between successive clusters.
   - `--keep-noise`: if set, keeps chains that are purely Gamma-Gamma (by default those are removed).

   ```bash
   uv run radiotrap analyze-sequences classification.csv chains.csv --radius 15.0 --time-window 1e9
   ```

#### Phase B: visualization (videos)

4. **Classified XY video** (Red = Alpha, Cyan = Beta, Green = Gamma, Gray = Other):

   - `--bin-size [FLOAT]`: time per video frame (in the same time units as the input, typically ns).

   ```bash
   uv run radiotrap render-classification segmented.txt classification.csv class_xy.mp4 --bin-size 100
   ```

5. **Side-view YT video** (time vs Y, scanning over X):

   ```bash
   uv run radiotrap render-yt segmented.txt classification.csv class_yt.mp4 --bin-size 1000
   ```

6. **Segmentation QC video** (random colors per cluster):

   ```bash
   uv run radiotrap render-segmentation segmented.txt seg_xy.mp4 --bin-size 100
   ```

#### Phase C: HTML reports

7. **Classification report**:

   - Open `report/report.html` in a browser.
   - Click **“Load CSV”** and select `classification.csv`.

8. **Sequence report**:

   - Open `report/sequence_report.html` in a browser.
   - Click **“Load CSV”** and select `chains.csv` from `analyze-sequences`.

---

### 4. Command reference

All usages below are invoked as `uv run radiotrap <command> ...`.

#### 4.1 `render`

Render raw events (no clustering) to MP4 or TIFF.

- **Usage:** `uv run radiotrap render INPUT_FILE OUTPUT_FILE`
- **Options:**
  - `--bin-size [FLOAT]` (default: `1e3`): time bin size.
  - `--fps [INT]` (default: `30`): frames per second for MP4.
  - `--format [mp4|tiff]` (default: `mp4`).
  - `--start-row [INT]` (default: `0`): first input row to load.
  - `--n-rows [INT]` (default: all): number of rows to load.

#### 4.2 `segment`

Group raw events into spatio-temporal clusters and write `Cluster_ID` to a text file.

- **Usage:** `uv run radiotrap segment INPUT_FILE OUTPUT_TXT --time-window ...`
- **Options:**
  - `--time-window [FLOAT]` (required): maximum time gap **in nanoseconds** between pixels in the same cluster.
  - `--spatial-radius [INT]` (default: `1`): spatial search radius in pixels.
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).

#### 4.3 `classify`

Classify clusters and compute per-cluster statistics.

- **Usage:** `uv run radiotrap classify SEGMENTED_TXT OUTPUT_CSV`
- **Options:**
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).
- **Output CSV:** one row per `Cluster_ID` with columns including:
  - `class` (Alpha / Beta / Gamma / Other),
  - timing (`mean_t`),
  - energy (`total_energy`, `mean_energy`),
  - shape (`mask_area`, `mask_roundness`, `max_radius`),
  - geometry (`min_x`, `max_x`, `min_y`, `max_y`, `width`, `height`, `mean_x`, `mean_y`),
  - `thumbnail_png` (base64 PNG thumbnail).

#### 4.4 `render-segmentation`

XY video with a random color per cluster (for checking segmentation).

- **Usage:** `uv run radiotrap render-segmentation SEGMENTED_TXT OUTPUT_MP4 --bin-size ...`
- **Options:**
  - `--bin-size [FLOAT]` (required): time per frame.
  - `--fps [INT]` (default: `30`).
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).

#### 4.5 `render-classification`

XY video with colors determined by the classification CSV.

- **Usage:** `uv run radiotrap render-classification SEGMENTED_TXT CLASS_CSV OUTPUT_MP4 --bin-size ...`
- **Options:**
  - `--bin-size [FLOAT]` (required).
  - `--fps [INT]` (default: `30`).
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).

#### 4.6 `render-yt`

Side-view YT video (time vs Y, scanning through X).

- **Usage:** `uv run radiotrap render-yt SEGMENTED_TXT CLASS_CSV OUTPUT_MP4 --bin-size ...`
- **Options:**
  - `--bin-size [FLOAT]` (required).
  - `--fps [INT]` (default: `30`).
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).

#### 4.7 `render-segmentation-yt`

Side-view YT video with random colors per cluster (segmentation only).

- **Usage:** `uv run radiotrap render-segmentation-yt SEGMENTED_TXT OUTPUT_MP4 --bin-size ...`
- **Options:**
  - `--bin-size [FLOAT]` (required).
  - `--fps [INT]` (default: `30`).
  - `--start-row [INT]` (default: `0`).
  - `--n-rows [INT]` (default: all).

#### 4.8 `analyze-sequences`

Find chains of clusters (A → B → C …) in space and time.

- **Usage:** `uv run radiotrap analyze-sequences CLASS_CSV CHAINS_CSV [options]`
- **Options:**
  - `--radius [FLOAT]` (default: `15.0`): spatial radius in pixels.
  - `--time-window [FLOAT]` (default: `1e9`): maximum allowed gap between events in **nanoseconds**.
  - `--keep-noise`: if set, keeps chains where all events are classified as Gamma; by default such chains are dropped.
- **Output CSV (`CHAINS_CSV`):**
  - One row per cluster in a chain with fields such as `Chain_ID`, `Sequence_Index`, `Cluster_ID`, `Class`, `Energy`, `Time_Abs`, `Time_Delta`, `Dist_Delta`, and selected shape/thumbnail information.

---

### 5. Working with large files

Several commands support partial loading of very large input files:

- **Commands with `--start-row` / `--n-rows`:**
  - `render`, `segment`, `classify`,
  - `render-segmentation`, `render-classification`,
  - `render-yt`, `render-segmentation-yt`.

Example: render only the first 10,000 events to check settings:

```bash
uv run radiotrap render input.txt test.mp4 --n-rows 10000
```

Use the same pattern with `segment` and the rendering commands when testing on subsets of large datasets.
