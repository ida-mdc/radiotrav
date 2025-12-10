# RadioTrap: Radiation Event Processor

RadioTrap is a tool for processing, analyzing, and visualizing radiation data from **Timepix3** detectors (files like `.t3pa` or `.txt`).

It allows you to:

1.  **Render** raw data into videos (MP4) or scientific images (TIFF).
2.  **Segment** individual particle tracks (clustering).
3.  **Classify** particles into Alpha, Beta, and Gamma radiation based on shape and energy.
4.  **Visualize** the results in 3D (XY views) and X-slicing views (Time vs Y).
5.  **Generate Reports** with detailed statistics and galleries.

-----

## 1\. Installation (First Time Setup)

We use a modern tool called **uv** to handle the Python setup automatically.

### Step 1: Install `uv`

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and paste this command:

**MacOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

*Close and reopen your terminal window after installing to ensure it works.*

### Step 2: Get the Code

Navigate to where you want to save the project and clone it:

```bash
git clone https://gitlab.com/ida-mdc/radiotrap.git
cd radiotrap
```

### Step 3: Install Dependencies

Run these two commands inside the `radiotrap` folder. This sets up a virtual environment and installs all necessary libraries (like pandas, opencv, etc.).

```bash
uv sync
uv pip install -e .
```

*You are now ready to run the software\!*

-----

## 2\. Quick Start: The Full Pipeline

Here is the standard workflow to go from raw text data to a fully classified video report.

**Note:** You must always type `uv run radiotrap` to execute commands.

### Phase A: Processing

1.  **Segment** the raw pixels into clusters (groups of connected pixels).

      * `--time-window`: Max time gap (ns) to link pixels.
      * `--spatial-radius`: Max pixel distance (1 or 2) to link pixels.

    <!-- end list -->

    ```bash
    uv run radiotrap segment input_data.txt segmented.txt --time-window 100 --spatial-radius 2
    ```

2.  **Classify** the clusters into Alpha/Beta/Gamma.

    ```bash
    uv run radiotrap classify segmented.txt classification.csv
    ```

### Phase B: Visualization (Videos)

3.  **Create a Classified Video** (Red=Alpha, Blue=Beta, Green=Gamma).

      * `--scale 4`: Makes the video 4x larger (1024px) so text is readable.
      * `--bin-size 100`: Accumulates 100ns of data per video frame.

    <!-- end list -->

    ```bash
    uv run radiotrap render-classification segmented.txt classification.csv output_class.mp4 --bin-size 100 --scale 4
    ```

4.  **Create a Side-View (YT) Video** (Time is width, Y is height).

    ```bash
    uv run radiotrap render-yt segmented.txt classification.csv output_yt.mp4 --bin-size 1000 --scale 2
    ```

### Phase C: Reporting

5.  Open the file `radiation_report.html` in Chrome or Firefox.
6.  Click **"Load CSV"** and select the `classification.csv` file generated in Step 2.
7.  You will see interactive charts, heatmaps, and a gallery of particle thumbnails.

-----

## 3\. Commands Reference

### `render`

Creates a raw visualization of the input file without any clustering.

  * **Usage:** `uv run radiotrap render [INPUT] [OUTPUT]`
  * **Options:**
      * `--format mp4` (default) or `tiff` (for scientific analysis).
      * `--bin-size`: Time per frame.

### `segment`

Groups raw events into clusters.

  * **Usage:** `uv run radiotrap segment [INPUT_TXT] [OUTPUT_TXT]`
  * **Critical Options:**
      * `--time-window [FLOAT]`: The most important setting. If your tracks look broken, increase this (e.g., 200). If distinct particles are merging, decrease it.
      * `--spatial-radius [INT]`: Use `1` for tight clusters, `2` if your sensor has dead pixels or gaps.

### `classify`

Analyzes shapes (roundness, size, energy) to determine particle type.

  * **Usage:** `uv run radiotrap classify [SEGMENTED_TXT] [OUTPUT_CSV]`
  * **Output:** Creates a CSV with columns for `class`, `energy`, `roundness`, and `thumbnail` images.

### `render-segmentation`

Renders a video where **each cluster has a random color**. Great for checking if your segmentation settings (`time-window`) are working correctly.

  * **Usage:** `uv run radiotrap render-segmentation [SEGMENTED_TXT] [OUTPUT_MP4]`

### `render-classification`

Renders a video colored by physics type (Alpha=Red, Beta=Blue, Gamma=Green).

  * **Usage:** `uv run radiotrap render-classification [SEGMENTED_TXT] [CLASS_CSV] [OUTPUT_MP4]`

### `render-yt`

Renders a side view (TY). It scans through the sensor from Left to Right (X=0 to 255).

  * **Image Width** = Time.
  * **Image Height** = Sensor Y position.
  * **Usage:** `uv run radiotrap render-yt [SEGMENTED_TXT] [CLASS_CSV] [OUTPUT_MP4]`

-----

## 4\. Useful Options for All Commands

You can add these flags to almost any command to tweak the output:

  * **`--scale [INT]`**

      * Upscales the video. Default is `4`.
      * Use `1` for raw 256x256 pixel output.
      * Use `4` or `8` for high-quality videos for presentations.

  * **`--start-row [INT]` and `--n-rows [INT]`**

      * **Lifesaver for huge files.** Instead of processing a 50GB file, you can process just a small chunk to test your settings.
      * Example: Process only the first 10,000 events:
        ```bash
        uv run radiotrap render input.txt test.mp4 --n-rows 10000
        ```

  * **`--fps [INT]`**

      * Frames per second of the output video. Default `30`.
