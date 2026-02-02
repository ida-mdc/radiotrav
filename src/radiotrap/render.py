import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from imageio import get_writer
import tifffile

from radiotrap.overlay import draw_hud


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _calc_time_params(t, bin_size, t_limits=None):
    if t_limits is not None:
        tmin, tmax = t_limits
    else:
        if len(t) == 0:
            raise ValueError("Cannot render: no events to render. Check your filters or input data.")
        tmin, tmax = t[0], t[-1]

    n_frames = int(np.ceil((tmax - tmin) / bin_size))
    n_frames = max(1, n_frames)
    if n_frames % 2 != 0: n_frames += 1

    return tmin, tmax, n_frames


def _get_discrete_colors(ids, color_lookup=None):
    if color_lookup is not None:
        safe_ids = np.clip(ids, 0, color_lookup.shape[0] - 1)
        return color_lookup[safe_ids]
    else:
        r = (ids * 53423423) % 256
        g = (ids * 94235252) % 256
        b = (ids * 19283741) % 256
        return np.column_stack((r, g, b)).astype(np.uint8)


def _format_time(ns):
    """
    Converts nanoseconds to human readable H:M:S.ms string.
    Format: HH:MM:SS.mmm
    """
    seconds = ns / 1e9

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _alpha_fade(t_event, t_center, half_window_ns):
    """
    Alpha for fading: fully opaque only at the current time point (t_center),
    linear fade to 0 at both edges of the time window.
    half_window_ns: half of the fade window in nanoseconds.
    Returns array same shape as t_event, values in [0, 1].
    """
    t_event = np.asarray(t_event)
    if half_window_ns <= 0:
        return np.ones_like(t_event, dtype=np.float64)
    d = np.abs(t_event - t_center) / half_window_ns
    return (1.0 - np.clip(d, 0.0, 1.0)).astype(np.float64)


# ============================================================
# 1. HEATMAP RENDERER (XY)
# ============================================================

def render_video_stream(t, x, y, tot, output_file, bin_size, fps, time_window_ns=None):
    tmin, tmax, n_frames = _calc_time_params(t, bin_size)
    BATCH_FRAMES = 100
    use_fade = time_window_ns is not None and time_window_ns > 0
    half_win_ns = float(time_window_ns) / 2.0 if use_fade else None
    print(f"Rendering Heatmap: {n_frames} frames..." + (" (fade window {} ns)".format(time_window_ns) if use_fade else ""))

    cmap = cm.get_cmap("turbo")
    max_energy = np.max(tot) if tot.size > 0 else 1.0

    with get_writer(output_file, fps=fps, format="ffmpeg", codec="libx264",
                    ffmpeg_params=["-qp", "0"], pixelformat="yuv420p") as w:

        for start_f in tqdm(range(0, n_frames, BATCH_FRAMES), desc="Encoding"):
            end_f = min(start_f + BATCH_FRAMES, n_frames)
            current_batch_size = end_f - start_f
            t_start = tmin + start_f * bin_size
            t_end = tmin + end_f * bin_size

            if use_fade:
                rgb = np.zeros((current_batch_size, 256, 256, 3), dtype=np.uint8)
                energy_acc = np.zeros((current_batch_size, 256, 256), dtype=np.float64)
                alpha_acc = np.zeros((current_batch_size, 256, 256), dtype=np.float64)
                for i in range(current_batch_size):
                    t_center = tmin + (start_f + i + 0.5) * bin_size
                    t_lo = t_center - half_win_ns
                    t_hi = t_center + half_win_ns
                    idx_lo = np.searchsorted(t, t_lo)
                    idx_hi = np.searchsorted(t, t_hi)
                    if idx_lo >= idx_hi:
                        continue
                    t_chunk = t[idx_lo:idx_hi]
                    x_chunk = np.clip(x[idx_lo:idx_hi].astype(np.int32), 0, 255)
                    y_chunk = np.clip(y[idx_lo:idx_hi].astype(np.int32), 0, 255)
                    tot_chunk = tot[idx_lo:idx_hi].astype(np.float64)
                    alpha = _alpha_fade(t_chunk, t_center, half_win_ns)
                    # Accumulate energy (sum all events at each pixel within the time window)
                    np.add.at(energy_acc[i], (y_chunk, x_chunk), tot_chunk * alpha)
                    np.add.at(alpha_acc[i], (y_chunk, x_chunk), alpha)
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Weighted average: total energy / total alpha (but we want sum, so use energy_acc directly)
                    # Actually, for energy visualization, we want the accumulated sum, not average
                    # So normalize by max_energy but keep the accumulated values
                    norm_energy = np.clip(energy_acc / max_energy, 0, 1)
                rgb_float = cmap(norm_energy)[..., :3]
                max_alpha = alpha_acc.max(axis=(1, 2), keepdims=True)
                max_alpha = np.where(max_alpha > 0, max_alpha, 1.0)
                brightness = np.clip(alpha_acc / max_alpha, 0, 1)
                rgb_float = rgb_float * brightness[..., np.newaxis]
                rgb = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
            else:
                idx_start = np.searchsorted(t, t_start)
                idx_end = np.searchsorted(t, t_end)
                t_chunk = t[idx_start:idx_end]
                x_chunk = x[idx_start:idx_end]
                y_chunk = y[idx_start:idx_end]
                tot_chunk = tot[idx_start:idx_end]

                if len(t_chunk) == 0:
                    rgb = np.zeros((current_batch_size, 256, 256, 3), dtype=np.uint8)
                else:
                    # Hard bins: sum energy at each pixel per frame
                    frame_indices = ((t_chunk - t_start) / bin_size).astype(np.int64)
                    frame_indices = np.clip(frame_indices, 0, current_batch_size - 1)
                    flat_indices = frame_indices * 65536 + y_chunk * 256 + x_chunk
                    total_voxels = current_batch_size * 65536

                    hits_flat = np.bincount(flat_indices, minlength=total_voxels)
                    energy_flat = np.bincount(flat_indices, weights=tot_chunk, minlength=total_voxels)

                    hits_3d = hits_flat.reshape((current_batch_size, 256, 256))
                    energy_3d = energy_flat.reshape((current_batch_size, 256, 256))

                    # Normalize accumulated energy sum
                    norm_energy = energy_3d / max_energy
                    rgb = cmap(norm_energy)[..., :3]

                    max_hits = hits_3d.max(axis=(1, 2), keepdims=True)
                    max_hits[max_hits == 0] = 1
                    brightness = hits_3d / max_hits
                    rgb = rgb * brightness[..., None]
                    rgb[hits_3d == 0] = 0
                    rgb = (rgb * 255).astype(np.uint8)

            for i in range(current_batch_size):
                curr_t = t_start + i * bin_size
                info = [f"T+ {_format_time(curr_t)}"]
                w.append_data(draw_hud(rgb[i], info))


# ============================================================
# 2. DISCRETE RENDERER (XY)
# ============================================================

def render_xy_discrete(t, x, y, cluster_ids, output_file, bin_size, fps, color_lookup=None, time_window_ns=None):
    tmin, tmax, n_frames = _calc_time_params(t, bin_size)
    BATCH_FRAMES = 100
    use_fade = time_window_ns is not None and time_window_ns > 0
    half_win_ns = float(time_window_ns) / 2.0 if use_fade else None
    print(f"Rendering Discrete XY: {n_frames} frames..." + (" (fade window {} ns)".format(time_window_ns) if use_fade else ""))

    mode_name = "classification" if color_lookup is not None else "segmentation"

    with get_writer(output_file, fps=fps, format="ffmpeg", codec="libx264",
                    ffmpeg_params=["-qp", "0"], pixelformat="yuv420p") as w:

        for start_f in tqdm(range(0, n_frames, BATCH_FRAMES), desc=f"Encoding {mode_name}"):
            end_f = min(start_f + BATCH_FRAMES, n_frames)
            current_batch_size = end_f - start_f
            t_start = tmin + start_f * bin_size
            t_end = tmin + end_f * bin_size

            if use_fade:
                # Sliding window + alpha fade: output = sum(color * alpha), so alpha actually fades
                # (no division by acc_alpha; otherwise a single event would always show full brightness)
                video_batch = np.zeros((current_batch_size, 256, 256, 3), dtype=np.float64)
                for i in range(current_batch_size):
                    t_center = tmin + (start_f + i + 0.5) * bin_size
                    t_lo = t_center - half_win_ns
                    t_hi = t_center + half_win_ns
                    idx_lo = np.searchsorted(t, t_lo)
                    idx_hi = np.searchsorted(t, t_hi)
                    if idx_lo >= idx_hi:
                        continue
                    t_chunk = t[idx_lo:idx_hi]
                    x_chunk = np.clip(x[idx_lo:idx_hi].astype(np.int32), 0, 255)
                    y_chunk = np.clip(y[idx_lo:idx_hi].astype(np.int32), 0, 255)
                    ids_chunk = cluster_ids[idx_lo:idx_hi]
                    alpha = _alpha_fade(t_chunk, t_center, half_win_ns)
                    colors = _get_discrete_colors(ids_chunk, color_lookup).astype(np.float64)
                    np.add.at(video_batch[i], (y_chunk, x_chunk), colors * alpha[:, np.newaxis])
                video_batch = np.clip(video_batch, 0, 255).astype(np.uint8)
            else:
                # Hard bins (original behavior)
                idx_start = np.searchsorted(t, t_start)
                idx_end = np.searchsorted(t, t_end)
                video_batch = np.zeros((current_batch_size, 256, 256, 3), dtype=np.uint8)
                x_chunk = x[idx_start:idx_end]
                if len(x_chunk) > 0:
                    y_chunk = y[idx_start:idx_end]
                    ids_chunk = cluster_ids[idx_start:idx_end]
                    t_rel = t[idx_start:idx_end] - t_start
                    frame_indices = (t_rel / bin_size).astype(np.int64)
                    frame_indices = np.clip(frame_indices, 0, current_batch_size - 1)
                    colors = _get_discrete_colors(ids_chunk, color_lookup)
                    video_batch[frame_indices, y_chunk, x_chunk] = colors

            for i in range(current_batch_size):
                curr_t = t_start + i * bin_size
                info = [f"T+ {_format_time(curr_t)}"]
                w.append_data(draw_hud(video_batch[i], info))


# ============================================================
# 3. DISCRETE YT RENDERER (Side View)
# ============================================================

def render_yt_discrete(t, x, y, cluster_ids, output_file, bin_size, fps, color_lookup=None, t_limits=None):
    """
    Renders YT-Plane (Scanning X).
    HUD shows the Time Range of the view.
    """
    tmin, tmax, n_time_bins = _calc_time_params(t, bin_size, t_limits)

    mode_name = "Class" if color_lookup is not None else "Seg"
    print(f"Rendering {mode_name} (YT): {n_time_bins}W x 256H...")

    if n_time_bins > 16384:
        print("WARNING: Video width > 16k pixels. FFmpeg might fail.")

    all_t_bins = ((t - tmin) / bin_size).astype(np.int64)
    all_t_bins = np.clip(all_t_bins, 0, n_time_bins - 1)

    with get_writer(output_file, fps=fps, format="ffmpeg", codec="libx264",
                    ffmpeg_params=["-qp", "0"], pixelformat="yuv420p") as w:

        for current_x in tqdm(range(256), desc="Encoding X-Slices"):
            frame = np.zeros((256, n_time_bins, 3), dtype=np.uint8)
            mask = (x == current_x)

            if np.any(mask):
                valid_t_bins = all_t_bins[mask]
                valid_y = y[mask]
                valid_ids = cluster_ids[mask]
                colors = _get_discrete_colors(valid_ids, color_lookup)
                frame[valid_y, valid_t_bins] = colors

            # HUD for YT:
            # Shows: "Slice X: 50"
            # Shows: "T-Range: 00:00:00 -> 01:00:00"
            info = [
                f"{_format_time(tmin)} -> {_format_time(tmax)}, slice X: {current_x}"
            ]
            w.append_data(draw_hud(frame, info))


# ============================================================
# 4. TIFF RENDERER
# ============================================================

def render_tiff_stream(t, x, y, output_file, bin_size):
    """
    Renders 3D hit volume to TIFF (uint16).
    """
    tmin, tmax, n_frames = _calc_time_params(t, bin_size)
    BATCH_FRAMES = 500

    print(f"Saving {n_frames} frames to TIFF...")

    with tifffile.TiffWriter(output_file, bigtiff=True) as tif:
        for start_f in tqdm(range(0, n_frames, BATCH_FRAMES), desc="TIFF"):
            end_f = min(start_f + BATCH_FRAMES, n_frames)
            current_batch_size = end_f - start_f
            t_start = tmin + start_f * bin_size
            t_end = tmin + end_f * bin_size

            idx_start = np.searchsorted(t, t_start)
            idx_end = np.searchsorted(t, t_end)

            hits_3d = np.zeros((current_batch_size, 256, 256), dtype=np.uint16)

            t_chunk = t[idx_start:idx_end]
            if len(t_chunk) > 0:
                x_chunk = x[idx_start:idx_end]
                y_chunk = y[idx_start:idx_end]

                frame_indices = ((t_chunk - t_start) / bin_size).astype(np.int64)
                frame_indices = np.clip(frame_indices, 0, current_batch_size - 1)

                flat_indices = frame_indices * 65536 + y_chunk * 256 + x_chunk
                hits_flat = np.bincount(flat_indices, minlength=current_batch_size * 65536)
                hits_3d = hits_flat.reshape((current_batch_size, 256, 256)).astype(np.uint16)

            for i in range(current_batch_size):
                tif.write(hits_3d[i], contiguous=True)