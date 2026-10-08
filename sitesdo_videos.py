#!/usr/bin/env python3
"""
video_auto_editor.py  (v6: optimize first, analyse once, cut the clips)
=======================================================================

For every input video this script now works in THREE clearly separated stages:

  STAGE 1  OPTIMIZE THE WHOLE VIDEO ONCE
           The original is encoded a single time into a clean "master"
           (<= --max-dim, even size, constant quality, short keyframe interval,
           faststart).  The original is never touched again.

  STAGE 2  ANALYSE THE MASTER ONCE (never the individual clips)
           * intro / outro detection (edges of the whole video),
           * a one-time refinement check of the cut points,
           * watermark / logo / credit-text detection over the whole usable
             window, then ONE calibration render that picks the removal-box
             size that really removes the overlay.
           All of it is duration-aware, so 5-20 second videos are analysed
           with the same care as 20-minute ones.

  STAGE 3  CUT THE CLIPS FROM THE MASTER
           The usable window is split into N-second parts (default 60, any
           value works, e.g. 10 or even 3).  Every part gets exactly the same
           trim + watermark removal that was decided in stage 2, so a 3 second
           part is treated identically to a 60 second one.  Nothing is
           re-detected per part.

Why this fixes "watermark / logo / intro / outro sometimes not removed":
  * Detection used to run on the raw download (odd sizes, rotation flags, VFR,
    broken seek tables) and was repeated on tiny parts.  The master is clean and
    seekable, and detection sees the whole video's motion.
  * Windows, sample spacing and "static card" tests were tuned for long videos.
    They now scale with the video length.
  * If ffmpeg's delogo cannot take a box, the region is now covered by a blur
    patch instead of silently leaving the logo on screen.

Quick use
---------
    python video_auto_editor.py --mode batch --batch ./in --outdir ./out --clip-seconds 10
    python video_auto_editor.py --input movie.mp4 --output out.mp4 --clip-seconds 3

Debug helpers: --debug-detect, --debug-preview, --watermark-box-pct x,y,w,h,
--intro-sec, --outro-sec, --no-master (analyse + cut the original directly).

Requirements: pip install opencv-python-headless numpy imagehash pillow ; ffmpeg on PATH.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Missing dependency: pip install opencv-python-headless")


# --------------------------------------------------------------------------- #
# Output / master settings
# --------------------------------------------------------------------------- #
MAX_DIM = 1920
VIDEO_CRF = 20
VIDEO_BITRATE = "6M"
AUDIO_BITRATE = "128k"
DEFAULT_ENCODE_PRESET = "medium"

# Stage-1 master: high quality + fast encode, so the second (final) encode of the
# clips loses almost nothing. Short GOP = accurate, fast seeking for analysis and cutting.
MASTER_CRF = 17
MASTER_PRESET = "veryfast"
MASTER_MAXRATE = "10M"
MASTER_GOP = 24

# Watermark detection runs on frames downscaled to at most this width.
DETECT_MAX_WIDTH = 960


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

#: Per-video notes collected while processing (reasons for skips / fallbacks /
#: undetected watermarks). Printed in the end-of-run report.
CURRENT_NOTES: List[str] = []
REPORT: List[dict] = []


def _warn(msg: str) -> None:
    print(f"  !! {msg}")
    CURRENT_NOTES.append(msg)


def run(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              errors="replace", stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n\nSTDERR:\n{proc.stderr}")
    return proc


def ffprobe_duration(path: str) -> float:
    """Container duration, falling back to the longest stream duration."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=duration",
           "-of", "json", path]
    data = json.loads(run(cmd, timeout=120).stdout)
    fmt = data.get("format", {}).get("duration")
    try:
        if fmt is not None and float(fmt) > 0:
            return float(fmt)
    except (TypeError, ValueError):
        pass
    best = 0.0
    for st in data.get("streams", []):
        try:
            best = max(best, float(st.get("duration")))
        except (TypeError, ValueError):
            continue
    if best <= 0:
        raise ValueError(f"Could not determine duration of {path}")
    return best


def ffprobe_dimensions(path: str) -> Tuple[int, int]:
    """Frame size AS FFMPEG'S FILTERS SEE IT (rotation flag applied)."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
           "-of", "json", path]
    out = json.loads(run(cmd, timeout=120).stdout)
    s = out["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = (s.get("tags") or {}).get("rotate")
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        if rot is not None and int(round(abs(float(rot)))) % 180 == 90:
            w, h = h, w
    except (TypeError, ValueError):
        pass
    return w, h


def is_readable_video(path: str) -> bool:
    try:
        w, h = ffprobe_dimensions(path)
        return w > 0 and h > 0 and ffprobe_duration(path) > 0.5
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# STAGE 1: optimized master of the WHOLE video
# --------------------------------------------------------------------------- #

def _scale_filters(frame_w: int, frame_h: int, optimize: bool, max_dim: int) -> List[str]:
    """Lanczos downscale ONLY when the source is bigger than max_dim, plus a
    guarantee of even dimensions (libx264/yuv420p refuse odd sizes)."""
    if optimize and max(frame_w, frame_h) > max_dim:
        if frame_w >= frame_h:
            return [f"scale={max_dim}:-2:flags=lanczos"]
        return [f"scale=-2:{max_dim}:flags=lanczos"]
    if frame_w % 2 or frame_h % 2:
        return ["scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"]
    return []


def make_optimized_master(src: str, dst: str, optimize: bool = True, max_dim: int = MAX_DIM,
                          audio_bitrate: str = AUDIO_BITRATE, crf: int = MASTER_CRF) -> str:
    """Encode the whole video once: size cap, even dims, rotation baked in, constant
    quality, short GOP, faststart. Everything after this reads the master."""
    src_dur = ffprobe_duration(src)
    fw, fh = ffprobe_dimensions(src)
    flt = _scale_filters(fw, fh, optimize, max_dim)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
           "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if flt:
        cmd += ["-vf", ",".join(flt)]
    cmd += ["-c:v", "libx264", "-preset", MASTER_PRESET, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", str(MASTER_GOP)]
    if optimize:
        cmd += ["-maxrate", MASTER_MAXRATE, "-bufsize", "20M"]
    cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2", "-movflags", "+faststart", dst]
    try:
        run(cmd, timeout=max(900, int(src_dur * 6)))
        if not os.path.exists(dst) or os.path.getsize(dst) < 2048:
            raise RuntimeError("master output missing or empty")
        m_dur = ffprobe_duration(dst)
        if m_dur < 0.9 * src_dur - 0.5:
            raise RuntimeError(f"master too short ({m_dur:.1f}s vs {src_dur:.1f}s)")
    except Exception:
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        raise
    return dst


def prepare_master(src: str, master_dir: str, idx: int, use_master: bool, optimize: bool,
                   max_dim: int, audio_bitrate: str, crf: int) -> Tuple[str, List[str]]:
    """Returns (path_to_analyse_and_cut, notes). Falls back to the original file if the
    master cannot be built, so a video is never lost because of this stage."""
    if not use_master:
        return src, []
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(src))[0])[:60] or "video"
    dst = os.path.join(master_dir, f"{idx:04d}_{stem}.mp4")
    t0 = time.time()
    try:
        make_optimized_master(src, dst, optimize=optimize, max_dim=max_dim,
                              audio_bitrate=audio_bitrate, crf=crf)
        w, h = ffprobe_dimensions(dst)
        print(f"  master  : {os.path.basename(src)} -> {w}x{h}, "
              f"{os.path.getsize(dst) / 1e6:.1f} MB in {time.time() - t0:.0f}s")
        return dst, []
    except Exception as e:   # noqa: BLE001
        msg = (f"optimized master failed ({str(e).strip()[-160:]}); "
               f"analysing and cutting the original instead")
        print(f"  !! {os.path.basename(src)}: {msg}")
        return src, [msg]


# --------------------------------------------------------------------------- #
# Frame sampling (duration-aware)
# --------------------------------------------------------------------------- #

def _adaptive_step(window: float, target_samples: int = 48, lo: float = 0.1, hi: float = 0.25) -> float:
    """Sample spacing for a scan window. Long windows keep the classic 0.25s; short
    windows are sampled more densely so a 2-4 second scan still has ~30-48 samples."""
    return float(np.clip(window / float(max(1, target_samples)), lo, hi))


def _sample_frame_stats(path: str, scan_seconds: float, step_sec: Optional[float] = None,
                        start_offset: float = 0.0):
    """Sample (absolute_t, mean_brightness, hist) every step_sec within
    [start_offset, start_offset + scan_seconds]."""
    if step_sec is None:
        step_sec = _adaptive_step(scan_seconds)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(step_sec * fps)))
    start_frame = int(start_offset * fps)
    stats = []
    frame_idx = start_frame
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t > start_offset + scan_seconds:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        stats.append((t, mean_b, hist))
        frame_idx += step_frames
    cap.release()
    return stats


# --------------------------------------------------------------------------- #
# Intro / outro detection -- single-video heuristics
# --------------------------------------------------------------------------- #

@dataclass
class BoundaryResult:
    time_sec: float          # intro: end-of-intro timestamp. outro: start-of-outro timestamp.
    method: str
    confidence: str          # "high" | "medium" | "low" | "none"


def _stats_step(stats) -> float:
    return max(1e-3, stats[1][0] - stats[0][0]) if len(stats) > 1 else 0.25


def _find_black_runs(stats, black_thresh=18.0, min_gap_frames=None):
    """Runs of near-black samples. The minimum run is ~0.4s of video whatever the
    sample spacing (2 samples at the classic 0.25s step)."""
    if min_gap_frames is None:
        min_gap_frames = max(2, int(round(0.4 / _stats_step(stats))))
    runs = []
    run_start = None
    for i, (t, mean_b, _h) in enumerate(stats):
        if mean_b < black_thresh:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= min_gap_frames:
                runs.append((stats[run_start][0], stats[i - 1][0]))
            run_start = None
    if run_start is not None and (len(stats) - run_start) >= min_gap_frames:
        runs.append((stats[run_start][0], stats[-1][0]))
    return runs


def _strongest_cuts(stats):
    cuts = []
    for i in range(1, len(stats)):
        h1, h2 = stats[i - 1][2], stats[i][2]
        score = float(cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA))
        cuts.append((stats[i][0], score))
    return cuts


DEFAULT_INTRO_MIN_SCAN_SEC = 15.0
DEFAULT_OUTRO_MIN_SCAN_SEC = 15.0
DEFAULT_OUTRO_SAFETY_MARGIN_SEC = 2.0


def _dominant_color_fraction(frame_bgr, sat_thresh: int = 60, hue_bin: int = 10) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, _v = cv2.split(hsv)
    mask = s > sat_thresh
    total = mask.size
    if mask.sum() < 0.05 * total:
        return 0.0
    hues = h[mask]
    hist = np.bincount((hues // hue_bin).astype(np.int32), minlength=(180 // hue_bin) + 1)
    return float(hist.max()) / total


def detect_color_bumper_intro(path: str, window_end: float, start_offset: float = 0.0,
                              dominance_thresh: float = 0.25, drop_frac: float = 0.5,
                              step_sec: Optional[float] = None, debug: bool = False) -> Optional[BoundaryResult]:
    """BoundaryResult if the video is on a solid-colour graphic bumper starting at
    `start_offset`, else None."""
    if window_end <= start_offset:
        return None
    if step_sec is None:
        step_sec = _adaptive_step(window_end - start_offset)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(step_sec * fps)))
    frame_idx = int(start_offset * fps)
    times, fracs = [], []
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t > window_end:
            break
        fracs.append(_dominant_color_fraction(frame))
        times.append(t)
        frame_idx += step_frames
    cap.release()

    if debug:
        print(f"  [intro-detect] color-bumper scan @ cursor={start_offset:.2f}s: dominant-hue fraction "
              f"per sample = {[round(f, 3) for f in fracs]} (need first sample > {dominance_thresh})")

    if not fracs or fracs[0] < dominance_thresh:
        if debug:
            print("  [intro-detect] frame at this cursor is not dominated by one solid color -- not a bumper")
        return None

    opening = fracs[0]
    end_t = times[-1]
    for t, f in zip(times, fracs):
        if f < opening * drop_frac:
            end_t = t
            break
    if debug:
        print(f"  [intro-detect] -> color/logo bumper (from cursor {start_offset:.2f}s) ends at {end_t:.2f}s")
    return BoundaryResult(round(end_t, 2), "color_bumper_intro", "high")


def _hist_distance(h1, h2) -> float:
    return float(cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA))


def _static_frames_needed(stats, stable_frames: Optional[int]) -> int:
    """Number of consecutive near-identical samples that make a 'static card'. Defined in
    TIME (~0.75s, i.e. 3 samples at the classic 0.25s step) so denser sampling of short
    videos does not turn ordinary slow footage into a false 'static intro/outro'."""
    if stable_frames is not None:
        return stable_frames
    return max(3, int(round(0.75 / _stats_step(stats))))


def _static_opening_boundary(stats, stable_frames: Optional[int] = None, stable_thresh: float = 0.15,
                             drift_thresh: float = 0.40, debug: bool = False) -> Optional[float]:
    if len(stats) < 3:
        return None
    stable_frames = _static_frames_needed(stats, stable_frames)
    if len(stats) < stable_frames + 2:
        return None
    pairwise = [_hist_distance(stats[i][2], stats[i + 1][2]) for i in range(stable_frames)]
    if debug:
        print(f"  [intro-detect] opening stability check, consecutive distances "
              f"(need ALL < {stable_thresh} to count as a static card): {[round(d, 3) for d in pairwise]}")
    if any(d > stable_thresh for d in pairwise):
        if debug:
            print("  [intro-detect] opening isn't static -- likely real footage already playing, skipping")
        return None

    ref_hist = stats[0][2]
    for t, _b, hist in stats[stable_frames:]:
        dist = _hist_distance(ref_hist, hist)
        if dist > drift_thresh:
            if debug:
                print(f"  [intro-detect] -> static opening drifts from its reference frame at "
                      f"{t:.2f}s (distance={dist:.3f} > {drift_thresh})")
            return t
    return None


def detect_intro_single(path: str, max_search_sec: float = DEFAULT_INTRO_MIN_SCAN_SEC,
                        debug: bool = False, cut_thresh: float = 0.35,
                        window_frac: float = 0.35, start_at: float = 0.0) -> BoundaryResult:
    """Intro detection starting at `start_at` (0 = beginning of the video).
    The scan window is min(max_search_sec, window_frac * remaining video), so a short
    video is never scanned almost end to end (the old rule looked at up to duration-1)."""
    duration = ffprobe_duration(path)
    remaining = max(0.0, duration - start_at)
    window = max(0.0, min(max_search_sec, window_frac * remaining, remaining - 1.0))
    scan_end = start_at + window
    stats = _sample_frame_stats(path, window, start_offset=start_at)
    if debug:
        print(f"  [intro-detect] scanning [{start_at:.2f}s, {scan_end:.2f}s] ({len(stats)} samples)")
    if len(stats) < 4:
        if debug:
            print(f"  [intro-detect] too few samples ({len(stats)}) -- insufficient data")
        return BoundaryResult(start_at, "insufficient_data", "none")

    cursor = start_at
    chained_methods = []
    tail_margin = max(0.15, min(0.5, 0.1 * window))

    # Chained detection: black flash -> logo bumper -> static card ... keep advancing
    # a cursor until real, changing footage starts or the scan limit is hit.
    while cursor < scan_end - tail_margin:
        sub_stats = [s for s in stats if s[0] >= cursor - 1e-6]
        if len(sub_stats) < 4:
            break
        if debug:
            print(f"  [intro-detect] --- cursor at {cursor:.2f}s, brightness={sub_stats[0][1]:.1f} ---")

        advanced_to = None
        method_used = None

        if sub_stats[0][1] < 18.0:
            black_runs = _find_black_runs(sub_stats)
            opening_run = black_runs[0] if black_runs and black_runs[0][0] < sub_stats[0][0] + 1.0 else None
            if opening_run is not None:
                run_end = opening_run[1]
                next_samples = [s for s in sub_stats if s[0] > run_end + 1e-6]
                advanced_to = next_samples[0][0] if next_samples else run_end
                method_used = "black_screen"
                if debug:
                    print(f"  [intro-detect] black screen at cursor -> run ends {run_end:.2f}s, "
                          f"advancing cursor to next frame at {advanced_to:.2f}s")

        if advanced_to is None:
            bumper = detect_color_bumper_intro(path, window_end=scan_end, start_offset=cursor, debug=debug)
            if bumper is not None:
                advanced_to = bumper.time_sec
                method_used = "color_bumper"

        if advanced_to is None:
            static_end = _static_opening_boundary(sub_stats, debug=debug)
            if static_end is not None:
                advanced_to = static_end
                method_used = "static_card"

        if advanced_to is None or advanced_to <= cursor + 1e-6:
            break

        chained_methods.append(method_used)
        cursor = advanced_to

    if chained_methods:
        method_str = "+".join(dict.fromkeys(chained_methods))
        if debug:
            print(f"  [intro-detect] -> chained detection [{method_str}], final intro end = {cursor:.2f}s")
        return BoundaryResult(round(cursor, 2), method_str, "high")

    cuts = _strongest_cuts(stats)
    if debug and cuts:
        top5 = sorted(cuts, key=lambda x: -x[1])[:5]
        print(f"  [intro-detect] top scene-cut candidates (time, score, need >{cut_thresh}): "
              f"{[(round(t, 2), round(s, 3)) for t, s in top5]}")
    if cuts:
        best_t, best_score = max(cuts, key=lambda x: x[1])
        if best_score > cut_thresh:
            if debug:
                print(f"  [intro-detect] -> using scene cut at {best_t:.2f}s (score={best_score:.3f})")
            return BoundaryResult(round(best_t, 2), "scene_cut", "medium")

    return BoundaryResult(start_at, "no_clear_boundary", "none")


# --------------------------------------------------------------------------- #
# Outro detection helpers
# --------------------------------------------------------------------------- #

def detect_color_bumper_outro(path: str, scan_start: float, scan_end: float,
                              dominance_thresh: float = 0.25, drop_frac: float = 0.5,
                              step_sec: Optional[float] = None, debug: bool = False) -> Optional[BoundaryResult]:
    if scan_end <= scan_start:
        return None
    if step_sec is None:
        step_sec = _adaptive_step(scan_end - scan_start)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(step_sec * fps)))
    frame_idx = int(scan_start * fps)
    times, fracs = [], []
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t > scan_end:
            break
        fracs.append(_dominant_color_fraction(frame))
        times.append(t)
        frame_idx += step_frames
    cap.release()

    if debug:
        print(f"  [outro-detect] color-bumper scan [{scan_start:.2f}s, {scan_end:.2f}s]: dominant-hue "
              f"fraction per sample = {[round(f, 3) for f in fracs]} (need last sample > {dominance_thresh})")

    if not fracs or fracs[-1] < dominance_thresh:
        if debug:
            print("  [outro-detect] final frame isn't dominated by one solid color -- not a color-bumper outro")
        return None

    ending_frac = fracs[-1]
    start_t = times[-1]
    for t, f in zip(times, fracs):
        if f >= ending_frac * drop_frac:
            start_t = t
            break
    if debug:
        print(f"  [outro-detect] -> color/logo bumper outro starts at {start_t:.2f}s")
    return BoundaryResult(round(start_t, 2), "color_bumper_outro", "high")


def _static_ending_boundary(stats, stable_frames: Optional[int] = None, stable_thresh: float = 0.15,
                            drift_thresh: float = 0.40, debug: bool = False) -> Optional[float]:
    if len(stats) < 3:
        return None
    stable_frames = _static_frames_needed(stats, stable_frames)
    if len(stats) < stable_frames + 2:
        return None
    tail = stats[-stable_frames:]
    pairwise = [_hist_distance(tail[i][2], tail[i + 1][2]) for i in range(len(tail) - 1)]
    if debug:
        print(f"  [outro-detect] ending stability check, consecutive distances "
              f"(need ALL < {stable_thresh} to count as a static end-card): {[round(d, 3) for d in pairwise]}")
    if any(d > stable_thresh for d in pairwise):
        if debug:
            print("  [outro-detect] ending isn't static -- likely real footage right up to EOF, skipping")
        return None

    ref_hist = stats[-1][2]
    boundary_t = stats[max(0, len(stats) - stable_frames)][0]
    for i in range(len(stats) - stable_frames - 1, -1, -1):
        t, _b, hist = stats[i]
        dist = _hist_distance(ref_hist, hist)
        if dist > drift_thresh:
            if debug:
                print(f"  [outro-detect] -> static ending drifts from its reference frame going back to "
                      f"{stats[i + 1][0]:.2f}s (distance={dist:.3f} > {drift_thresh})")
            return stats[i + 1][0]
        boundary_t = t
    return boundary_t


def _fade_to_black_boundary(stats, end_thresh: float = 30.0, drop_ratio: float = 0.6,
                            debug: bool = False) -> Optional[float]:
    if len(stats) < 4:
        return None
    final_b = stats[-1][1]
    if final_b > end_thresh:
        if debug:
            print(f"  [outro-detect] fade-to-black check: final brightness {final_b:.1f} is above "
                  f"the near-black threshold ({end_thresh}) -- video doesn't end dark, skipping")
        return None

    peak_b = max(b for _t, b, _h in stats)
    if peak_b <= 0:
        return None
    target = peak_b * (1.0 - drop_ratio)

    for t, b, _h in stats:
        if b <= target:
            if debug:
                print(f"  [outro-detect] -> brightness fade starts at {t:.2f}s "
                      f"(peak={peak_b:.1f}, dropped to {b:.1f} <= target {target:.1f})")
            return t
    return None


DEFAULT_OUTRO_MAX_SEARCH_SEC = DEFAULT_OUTRO_MIN_SCAN_SEC


def detect_outro_single(path: str, max_search_sec: float = DEFAULT_OUTRO_MAX_SEARCH_SEC,
                        safety_margin: float = DEFAULT_OUTRO_SAFETY_MARGIN_SEC,
                        debug: bool = False, cut_thresh: float = 0.35,
                        window_frac: float = 0.4, end_at: Optional[float] = None) -> BoundaryResult:
    """Scans the window that ENDS at `end_at` (default: end of the video) for the EARLIEST
    boundary of any outro pattern (black gap, colour end card, static end card, fade to
    black, strong scene cut). A safety margin is subtracted from the winner."""
    file_dur = ffprobe_duration(path)
    duration = file_dur if end_at is None else min(end_at, file_dur)
    window = min(max_search_sec, duration * window_frac)
    start_offset = max(0.0, duration - window)
    stats = _sample_frame_stats(path, window, start_offset=start_offset)
    if debug:
        print(f"  [outro-detect] scanning [{start_offset:.2f}s, {duration:.2f}s] ({len(stats)} samples)")
    if len(stats) < 4:
        if debug:
            print(f"  [outro-detect] too few samples ({len(stats)}) -- insufficient data")
        return BoundaryResult(duration, "insufficient_data", "none")

    end_guard = min(0.5, 0.1 * window)
    candidates: List[BoundaryResult] = []

    black_runs = _find_black_runs(stats)
    if debug:
        print(f"  [outro-detect] black runs found in window: "
              f"{[(round(s, 2), round(e, 2)) for s, e in black_runs]}")
    candidate_gaps = [g for g in black_runs if g[0] < duration - end_guard]
    if candidate_gaps:
        candidates.append(BoundaryResult(round(candidate_gaps[0][0], 2), "black_frame_gap", "high"))

    bumper = detect_color_bumper_outro(path, start_offset, duration, debug=debug)
    if bumper is not None:
        candidates.append(bumper)

    static_t = _static_ending_boundary(stats, debug=debug)
    if static_t is not None and static_t <= stats[0][0] + 0.3:
        if debug:
            print("  [outro-detect] entire scan window is 'static' -- no card start found, ignoring")
        static_t = None
    if static_t is not None:
        candidates.append(BoundaryResult(round(static_t, 2), "static_card_outro", "high"))

    fade_t = _fade_to_black_boundary(stats, debug=debug)
    if fade_t is not None:
        candidates.append(BoundaryResult(round(fade_t, 2), "fade_to_black_outro", "high"))

    if candidates:
        best = min(candidates, key=lambda c: c.time_sec)
        if debug:
            all_str = ", ".join(f"{c.method}={c.time_sec:.2f}s" for c in candidates)
            print(f"  [outro-detect] candidates found: [{all_str}] -> earliest is {best.method} "
                  f"@ {best.time_sec:.2f}s")
        margin_applied = min(safety_margin, max(0.0, best.time_sec - start_offset))
        final_t = max(start_offset, best.time_sec - margin_applied)
        method = best.method if margin_applied <= 0 else f"{best.method}_margin{margin_applied:.1f}s"
        return BoundaryResult(round(final_t, 2), method, best.confidence)

    cuts = _strongest_cuts(stats)
    strong_cuts = [c for c in cuts if c[1] > cut_thresh]
    if debug and cuts:
        top5 = sorted(cuts, key=lambda x: -x[1])[:5]
        print(f"  [outro-detect] top scene-cut candidates (time, score, need >{cut_thresh}): "
              f"{[(round(t, 2), round(s, 3)) for t, s in top5]}")
    if strong_cuts:
        earliest_t, _score = min(strong_cuts, key=lambda x: x[0])
        margin_applied = min(safety_margin, max(0.0, earliest_t - start_offset))
        final_t = max(start_offset, earliest_t - margin_applied)
        method = "scene_cut" if margin_applied <= 0 else f"scene_cut_margin{margin_applied:.1f}s"
        return BoundaryResult(round(final_t, 2), method, "medium")

    return BoundaryResult(duration, "no_clear_boundary", "none")


# Retry ladders: each attempt widens the scan window and relaxes the scene-cut threshold.
INTRO_RETRIES = [  # (window multiplier, scene-cut threshold, max share of the video scanned)
    (1.0, 0.35, 0.35), (1.7, 0.28, 0.45), (2.5, 0.22, 0.55),
]
OUTRO_RETRIES = [  # (window multiplier, scene-cut threshold, share of video, extra safety margin)
    (1.0, 0.35, 0.40, 0.0), (1.7, 0.28, 0.50, 0.5), (2.5, 0.22, 0.60, 1.0),
]


def detect_intro_with_retries(path: str, max_search_sec: float = DEFAULT_INTRO_MIN_SCAN_SEC,
                              debug: bool = False) -> BoundaryResult:
    last = BoundaryResult(0.0, "no_clear_boundary", "none")
    duration = ffprobe_duration(path)
    for n, (mult, cut, frac) in enumerate(INTRO_RETRIES, 1):
        try:
            r = detect_intro_single(path, max_search_sec=max_search_sec * mult, debug=debug,
                                    cut_thresh=cut, window_frac=frac)
        except Exception as e:   # noqa: BLE001
            if debug:
                print(f"  [intro-detect] attempt {n} crashed: {e}")
            continue
        if r.confidence != "none" and r.time_sec > 0.0:
            if n > 1 and r.time_sec > min(0.25 * duration, 60.0):
                if debug:
                    print(f"  [intro-detect] retry {n} result {r.time_sec:.1f}s exceeds the sanity cap -- ignored")
                last = r
                continue
            if n > 1:
                print(f"  intro found on retry {n}/{len(INTRO_RETRIES)} (window x{mult}, cut>{cut}): "
                      f"{r.time_sec:.2f}s via {r.method}")
            return r
        last = r
    return last


def detect_outro_with_retries(path: str, max_search_sec: float = DEFAULT_OUTRO_MAX_SEARCH_SEC,
                              safety_margin: float = DEFAULT_OUTRO_SAFETY_MARGIN_SEC,
                              debug: bool = False) -> BoundaryResult:
    duration = ffprobe_duration(path)
    last = BoundaryResult(duration, "no_clear_boundary", "none")
    for n, (mult, cut, frac, extra) in enumerate(OUTRO_RETRIES, 1):
        try:
            r = detect_outro_single(path, max_search_sec=max_search_sec * mult,
                                    safety_margin=safety_margin + extra, debug=debug,
                                    cut_thresh=cut, window_frac=frac)
        except Exception as e:   # noqa: BLE001
            if debug:
                print(f"  [outro-detect] attempt {n} crashed: {e}")
            continue
        if r.confidence != "none" and r.time_sec < duration - 0.3:
            if n > 1 and (duration - r.time_sec) > min(0.25 * duration, 60.0):
                if debug:
                    print(f"  [outro-detect] retry {n} would trim {duration - r.time_sec:.1f}s (sanity cap) -- ignored")
                last = r
                continue
            if n > 1:
                print(f"  outro found on retry {n}/{len(OUTRO_RETRIES)} (window x{mult}, cut>{cut}): "
                      f"starts {r.time_sec:.2f}s via {r.method}")
            return r
        last = r
    return last


def refine_boundaries_on_master(path: str, intro_end: float, outro_start: float,
                                check_intro: bool, check_outro: bool, debug: bool = False
                                ) -> Tuple[float, float]:
    """ONE check per video (on the master, before cutting): is there still a black /
    bumper / fade right after the chosen intro end or right before the chosen outro
    start? Returns (extra_seconds_to_cut_at_start, extra_seconds_to_cut_at_end).
    This replaces the old per-part re-check of the first and last rendered part."""
    usable = outro_start - intro_end
    if usable < 3.0:
        return 0.0, 0.0
    win = min(10.0, 0.35 * usable)
    cap = min(6.0, 0.25 * usable)
    lead = tail = 0.0
    if check_intro:
        try:
            r = detect_intro_single(path, max_search_sec=win, window_frac=1.0,
                                    start_at=intro_end, debug=debug)
            extra = r.time_sec - intro_end
            if (r.confidence == "high" and not r.method.startswith("static_card")
                    and 0.3 <= extra <= cap):
                lead = extra
        except Exception:   # noqa: BLE001
            pass
    if check_outro:
        try:
            r = detect_outro_single(path, max_search_sec=win, safety_margin=0.5,
                                    window_frac=1.0, end_at=outro_start, debug=debug)
            extra = outro_start - r.time_sec
            # static-card / scene-cut hits are ignored here: slow real footage looks "static"
            # and trimming good content is worse than leaving a frame or two.
            if (r.confidence == "high" and not r.method.startswith(("static_card", "scene_cut"))
                    and 0.5 <= extra <= cap):
                tail = extra
        except Exception:   # noqa: BLE001
            pass
    if usable - lead - tail < max(2.0, 0.5 * usable):
        return 0.0, 0.0
    return lead, tail


# --------------------------------------------------------------------------- #
# Intro/outro detection -- BATCH mode (shared across many videos)
# --------------------------------------------------------------------------- #

def _phash_sequence(path: str, scan_seconds: float, start_offset: float = 0.0,
                    fps_sample: float = 1.0) -> List[int]:
    import imagehash
    from PIL import Image

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(fps / fps_sample)))
    start_frame = int(start_offset * fps)
    hashes = []
    frame_idx = start_frame
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t > start_offset + scan_seconds:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        h = imagehash.phash(img)
        hashes.append(int(str(h), 16))
        frame_idx += step_frames
    cap.release()
    return hashes


def _batch_sampling(paths: List[str]) -> Tuple[float, int]:
    """(samples per second, minimum matching samples) for the shared intro/outro test.
    Batches containing short videos are sampled 4x/s so a 2-3 second shared bumper still
    produces enough matching samples (it needs ~1.5-2s of common material either way)."""
    shortest = min(ffprobe_duration(p) for p in paths)
    fps_sample = 4.0 if shortest < 40.0 else 1.0
    return fps_sample, max(2, int(round(1.5 * fps_sample)))


def detect_intro_batch(paths: List[str], scan_seconds: float = DEFAULT_INTRO_MIN_SCAN_SEC,
                       hamming_thresh: int = 8) -> BoundaryResult:
    fps_sample, min_samples = _batch_sampling(paths)
    shortest = min(ffprobe_duration(p) for p in paths)
    scan = min(scan_seconds, 0.35 * shortest)
    sequences = [_phash_sequence(p, scan, start_offset=0.0, fps_sample=fps_sample) for p in paths]
    min_len = min(len(s) for s in sequences)
    if min_len == 0:
        return BoundaryResult(0.0, "batch_no_frames", "none")

    matched = 0
    for i in range(min_len):
        ref = sequences[0][i]
        ok = all(bin(ref ^ seq[i]).count("1") <= hamming_thresh for seq in sequences[1:])
        if ok:
            matched = i + 1
        else:
            break
    confidence = "high" if matched >= min_samples else "none"
    return BoundaryResult(matched / fps_sample, "batch_common_prefix", confidence)


def detect_outro_batch(paths: List[str], scan_seconds: float = 30.0,
                       hamming_thresh: int = 8,
                       safety_margin: float = DEFAULT_OUTRO_SAFETY_MARGIN_SEC) -> Optional[float]:
    """Shared outro LENGTH (seconds from each video's own end) if every video shares a
    common trailing sequence, else None."""
    fps_sample, min_samples = _batch_sampling(paths)
    durations = [ffprobe_duration(p) for p in paths]
    seqs = []
    for p, d in zip(paths, durations):
        window = min(scan_seconds, d * 0.4)
        offset = max(0.0, d - window)
        seqs.append(_phash_sequence(p, window, start_offset=offset, fps_sample=fps_sample))
    min_len = min(len(s) for s in seqs)
    if min_len == 0:
        return None

    matched = 0
    for i in range(1, min_len + 1):
        ref = seqs[0][-i]
        ok = all(bin(ref ^ seq[-i]).count("1") <= hamming_thresh for seq in seqs[1:])
        if ok:
            matched = i
        else:
            break
    if matched < min_samples:
        return None
    return matched / fps_sample + max(0.0, safety_margin)


# --------------------------------------------------------------------------- #
# Watermark / logo / credit-text detection (runs ONCE per video, on the master)
# --------------------------------------------------------------------------- #

@dataclass
class WatermarkBox:
    x: int
    y: int
    w: int
    h: int
    conf: float = 0.0   # fraction of box pixels whose edges persist (0 = unknown/manual)

    def clamped(self, frame_w: int, frame_h: int, pad: int = 4, margin: int = 4,
                pad_y: Optional[int] = None) -> "WatermarkBox":
        """Padded copy that fits inside the frame with a safety margin (delogo needs real
        breathing room). Returns an INVALID box (0,0,0,0) when nothing sane is left."""
        x_min, y_min = margin, margin
        x_max, y_max = frame_w - margin, frame_h - margin
        if x_max <= x_min or y_max <= y_min:
            return WatermarkBox(0, 0, 0, 0)
        py = pad if pad_y is None else pad_y
        x1 = max(self.x - pad, x_min)
        y1 = max(self.y - py, y_min)
        x2 = min(self.x + self.w + pad, x_max)
        y2 = min(self.y + self.h + py, y_max)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            return WatermarkBox(0, 0, 0, 0)
        w = max(2, w - (w % 2))
        h = max(2, h - (h % 2))
        return WatermarkBox(x1, y1, w, h)

    def to_fractions(self, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        if frame_w <= 0 or frame_h <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (self.x / frame_w, self.y / frame_h, self.w / frame_w, self.h / frame_h)

    def as_ffmpeg_delogo(self) -> str:
        return f"delogo=x={self.x}:y={self.y}:w={self.w}:h={self.h}:show=0"

    def as_blur_patch(self, idx: int = 0) -> str:
        """Fallback remover that works at ANY position (delogo refuses boxes near the frame
        edge): blur just this rectangle and overlay it back. Valid as a -vf filtergraph."""
        r = max(1, min(self.w, self.h) // 4 - 1)   # chroma planes are half size: radius <= min(w,h)/4
        a, b, c = f"bpa{idx}", f"bpb{idx}", f"bpc{idx}"
        return (f"split[{a}][{b}];[{b}]crop={self.w}:{self.h}:{self.x}:{self.y},"
                f"boxblur={r}:3[{c}];[{a}][{c}]overlay={self.x}:{self.y}")


def watermark_box_from_fractions(xp: float, yp: float, wp: float, hp: float,
                                 frame_w: int, frame_h: int) -> WatermarkBox:
    return WatermarkBox(round(xp * frame_w), round(yp * frame_h), round(wp * frame_w), round(hp * frame_h))


def _boxes_overlap_or_close(a, b, gap: int) -> bool:
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax1 - gap > bx2 or bx1 - gap > ax2 or ay1 - gap > by2 or by1 - gap > ay2)


def _cluster_boxes(boxes, gap: int):
    clusters = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _boxes_overlap_or_close(tuple(cur), tuple(clusters[j]), gap):
                    x1 = min(cur[0], clusters[j][0])
                    y1 = min(cur[1], clusters[j][1])
                    x2 = max(cur[0] + cur[2], clusters[j][0] + clusters[j][2])
                    y2 = max(cur[1] + cur[3], clusters[j][1] + clusters[j][3])
                    cur = [x1, y1, x2 - x1, y2 - y1]
                    used[j] = True
                    merged = True
            out.append(cur)
            used[i] = True
        clusters = out
    return [tuple(c) for c in clusters]


def _otsu_threshold(arr: np.ndarray, bins: int = 256) -> float:
    scaled = np.clip(arr, 0.0, 1.0)
    hist, edges = np.histogram(scaled, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    sum_all = float(np.dot(hist, centers))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    best_thresh = float(centers[0])
    for i in range(bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += centers[i] * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = float(centers[i])
    return best_thresh


def _expand_box_by_density(mask01: np.ndarray, x: int, y: int, w: int, h: int,
                           min_density: float = 0.18, step: int = 2,
                           max_frame_frac: float = 0.22, debug: bool = False):
    H, W = mask01.shape
    frame_area = H * W
    orig = (x, y, w, h)

    def row_density(yy: int) -> float:
        yy0, yy1 = max(0, yy), min(H, yy + step)
        xx0, xx1 = max(0, x), min(W, x + w)
        if yy1 <= yy0 or xx1 <= xx0:
            return 0.0
        return float(mask01[yy0:yy1, xx0:xx1].mean())

    def col_density(xx: int) -> float:
        xx0, xx1 = max(0, xx), min(W, xx + step)
        yy0, yy1 = max(0, y), min(H, y + h)
        if xx1 <= xx0 or yy1 <= yy0:
            return 0.0
        return float(mask01[yy0:yy1, xx0:xx1].mean())

    for _ in range(400):
        if y <= 0 or (w * h) / frame_area > max_frame_frac:
            break
        if row_density(y - step) < min_density:
            break
        s = min(step, y)
        y -= s
        h += s
    for _ in range(400):
        if y + h >= H or (w * h) / frame_area > max_frame_frac:
            break
        if row_density(y + h) < min_density:
            break
        h += min(step, H - (y + h))
    for _ in range(400):
        if x <= 0 or (w * h) / frame_area > max_frame_frac:
            break
        if col_density(x - step) < min_density:
            break
        s = min(step, x)
        x -= s
        w += s
    for _ in range(400):
        if x + w >= W or (w * h) / frame_area > max_frame_frac:
            break
        if col_density(x + w) < min_density:
            break
        w += min(step, W - (x + w))

    if (w * h) / frame_area > max_frame_frac:
        if debug:
            print(f"  [watermark-detect] expansion ran away past {max_frame_frac:.0%} of frame -- reverting")
        return orig
    return x, y, w, h


def _is_bar_shape(x, y, w, h, frame_w, frame_h, bar_span_frac=0.75, bar_thin_frac=0.08) -> bool:
    spans_width = w >= bar_span_frac * frame_w
    spans_height = h >= bar_span_frac * frame_h
    thin_height = h <= bar_thin_frac * frame_h
    thin_width = w <= bar_thin_frac * frame_w
    return (spans_width and thin_height) or (spans_height and thin_width)


def _split_oversized_box(mask01, box, frame_w, frame_h, max_w_frac=0.55, max_h_frac=0.55):
    x, y, w, h = box
    if w < max_w_frac * frame_w and h < max_h_frac * frame_h:
        return [box]
    sub = mask01[y:y + h, x:x + w]
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    pieces = []
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < 6:
            continue
        pieces.append((x + int(sx), y + int(sy), int(sw), int(sh)))
    return pieces if pieces else [box]


def _corner_proximity_score(x, y, w, h, frame_w, frame_h) -> float:
    cx, cy = x + w / 2.0, y + h / 2.0
    corners = [(0, 0), (frame_w, 0), (0, frame_h), (frame_w, frame_h)]
    dist = min(((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5 for ccx, ccy in corners)
    diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
    return max(0.0, 1.0 - dist / (diag * 0.35))


def _regions_overlap_or_close_dynamic(a, b, min_gap: int = 18, gap_frac: float = 0.6) -> bool:
    a_scale = max(a[2], a[3])
    b_scale = max(b[2], b[3])
    gap = max(min_gap, int(round(gap_frac * min(a_scale, b_scale))))
    return _boxes_overlap_or_close(a, b, gap)


def _merge_nearby_regions(boxes):
    clusters = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False] * len(clusters)
        for i in range(len(clusters)):
            if used[i]:
                continue
            cur = clusters[i]
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _regions_overlap_or_close_dynamic(tuple(cur), tuple(clusters[j])):
                    x1 = min(cur[0], clusters[j][0])
                    y1 = min(cur[1], clusters[j][1])
                    x2 = max(cur[0] + cur[2], clusters[j][0] + clusters[j][2])
                    y2 = max(cur[1] + cur[3], clusters[j][1] + clusters[j][3])
                    cur = [x1, y1, x2 - x1, y2 - y1]
                    used[j] = True
                    merged = True
            out.append(cur)
            used[i] = True
        clusters = out
    return [tuple(c) for c in clusters]


_FRAME_CACHE: dict = {}


def _read_gray_frames(path: str, start_sec: float, end_sec: float, count: int):
    key = (path, round(start_sec, 2), round(end_sec, 2), count)
    if key not in _FRAME_CACHE:
        if len(_FRAME_CACHE) > 12:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[key] = _read_gray_frames_uncached(path, start_sec, end_sec, count)
    return _FRAME_CACHE[key]


def _read_gray_frames_uncached(path: str, start_sec: float, end_sec: float, count: int):
    """Sample `count` grayscale frames across [start_sec, end_sec]; ffmpeg fills in any
    timestamps OpenCV cannot seek to. Returns (frames, true_w, true_h, det_scale)."""
    frames: List[np.ndarray] = []
    true_w = true_h = 0
    det_scale = 1.0
    times = np.linspace(start_sec, max(start_sec + 0.05, end_sec), num=count)

    def _push(gray):
        nonlocal true_w, true_h, det_scale
        if not true_w:
            true_h, true_w = gray.shape
            if true_w > DETECT_MAX_WIDTH:
                det_scale = true_w / float(DETECT_MAX_WIDTH)
        if det_scale != 1.0:
            gray = cv2.resize(gray, (int(round(true_w / det_scale)), int(round(true_h / det_scale))),
                              interpolation=cv2.INTER_AREA)
        frames.append(gray)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    missing = []
    for t in times:
        idx = int(t * fps)
        if total_frames > 0:
            idx = min(idx, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            missing.append(float(t))
            continue
        _push(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(frames) < 0.6 * count and missing:
        for t in missing:
            try:
                proc = subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                     "-f", "image2pipe", "-vcodec", "png", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, timeout=60)
                if proc.returncode != 0 or not proc.stdout:
                    continue
                img = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if true_w and img.shape[::-1] != (true_w, true_h):
                    img = cv2.resize(img, (true_w, true_h), interpolation=cv2.INTER_AREA)
                _push(img)
            except Exception:   # noqa: BLE001
                continue
    return frames, true_w, true_h, det_scale


def _box_persistence(edge_freq: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[float, float]:
    """(static_frac, ring_ratio): share of box pixels with persistent edges, and mean edge
    frequency inside the box / just outside it."""
    H, W = edge_freq.shape
    inner = edge_freq[y:y + h, x:x + w]
    if inner.size == 0:
        return 0.0, 0.0
    static_frac = float((inner >= 0.4).mean())
    pad = max(4, int(0.25 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    outer = edge_freq[y0:y1, x0:x1]
    ring_sum = float(outer.sum() - inner.sum())
    ring_n = outer.size - inner.size
    ring_mean = ring_sum / ring_n if ring_n > 0 else 0.0
    return static_frac, float(inner.mean()) / (ring_mean + 1e-3)


def _detect_wm_pass(path: str, start_sec: float, end_sec: float, sample_count: int = 48,
                    min_area_frac: float = 0.0004, max_area_frac: float = 0.45,
                    density_floor: float = 0.06, max_regions: int = 3,
                    border_frac: float = 0.32, debug: bool = False,
                    peak_floor: float = 0.12, thresh_lo: float = 0.10,
                    min_ring_ratio: float = 1.4, min_static_frac: float = 0.02,
                    preloaded=None) -> List[WatermarkBox]:
    """One detection pass over the given frames: per-pixel opaque/transparent overlay score,
    Otsu cutoff, clustering, bar/density filtering, region growing, persistence validation."""
    if preloaded is not None:
        frames_gray, true_w, true_h, det_scale = preloaded
    else:
        frames_gray, true_w, true_h, det_scale = _read_gray_frames(path, start_sec, end_sec, sample_count)
    if len(frames_gray) < 6:
        if debug:
            print(f"  [watermark-detect] only {len(frames_gray)} frames could be read -- too few")
        return []

    med = float(np.median(np.stack([f[::8, ::8] for f in frames_gray])))
    c_lo = int(np.clip(0.66 * med, 25, 70))
    c_hi = int(np.clip(c_lo * 2.4, 70, 180))
    edge_accum = None
    for g in frames_gray:
        e = cv2.Canny(g, c_lo, c_hi).astype(np.float32)
        edge_accum = e if edge_accum is None else edge_accum + e

    n = len(frames_gray)
    h_frame, w_frame = frames_gray[0].shape
    frame_area = h_frame * w_frame
    edge_freq = edge_accum / (255.0 * n)

    stack = np.stack(frames_gray, axis=0).astype(np.float32)
    variance = stack.var(axis=0)
    var_norm = variance / (variance.max() + 1e-6)

    opaque_score = edge_freq * (1.0 - var_norm)
    border_px = border_frac * min(w_frame, h_frame)
    yy, xx = np.mgrid[0:h_frame, 0:w_frame]
    border_mask = ((xx < border_px) | (yy < border_px) |
                   (w_frame - xx <= border_px) | (h_frame - yy <= border_px))
    transparent_score = edge_freq * border_mask
    combined = np.maximum(opaque_score, transparent_score * 0.85)
    peak = float(combined.max())

    if debug:
        print(f"  [watermark-detect] sampled {n} frames over [{start_sec:.1f}s, {end_sec:.1f}s], "
              f"resolution={w_frame}x{h_frame} peak={peak:.3f} mean={combined.mean():.4f}")
    if peak < peak_floor:
        return []

    thresh = _otsu_threshold(combined)
    thresh = float(np.clip(thresh, thresh_lo, max(thresh_lo, peak * 0.65)))
    raw_mask01 = (combined >= thresh).astype(np.uint8)

    kx = max(9, int(round(w_frame / 130.0)))
    ky = max(3, int(round(h_frame / 220.0)))
    kx += (kx % 2 == 0)
    ky += (ky % 2 == 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(raw_mask01 * 255, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    stable_boxes, mergeable_boxes = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        was_oversized = (w >= 0.55 * w_frame) or (h >= 0.55 * h_frame)
        for (sx, sy, sw, sh) in _split_oversized_box(raw_mask01, (x, y, w, h), w_frame, h_frame):
            area_frac = (sw * sh) / frame_area
            if not (min_area_frac <= area_frac <= max_area_frac):
                continue
            (stable_boxes if was_oversized else mergeable_boxes).append((sx, sy, sw, sh))
    if not stable_boxes and not mergeable_boxes:
        return []

    merge_gap = max(12, int(round(min(w_frame, h_frame) / 45.0)))
    clustered = stable_boxes + _cluster_boxes(mergeable_boxes, gap=merge_gap)
    survivors = [c for c in clustered if not _is_bar_shape(*c, w_frame, h_frame)]

    dense_enough = []
    for (x, y, w, h) in survivors:
        if float(raw_mask01[y:y + h, x:x + w].mean()) >= density_floor:
            dense_enough.append((x, y, w, h))
    if not dense_enough:
        return []

    merged_regions = _merge_nearby_regions(dense_enough)
    merged_regions = [r for r in merged_regions if not _is_bar_shape(*r, w_frame, h_frame)]
    if not merged_regions:
        return []

    scored = []
    for (x, y, w, h) in merged_regions:
        area_frac = (w * h) / frame_area
        corner_score = _corner_proximity_score(x, y, w, h, w_frame, h_frame)
        density = float(raw_mask01[y:y + h, x:x + w].mean())
        compactness = max(0.0, 1.0 - area_frac / 0.5)
        score = density * 0.45 + corner_score * 0.30 + compactness * 0.25
        scored.append(((x, y, w, h), score, area_frac))
    scored.sort(key=lambda t: t[1], reverse=True)

    final_boxes = []
    for (x, y, w, h), _score, pre_area_frac in scored[:max_regions]:
        grow_cap = float(np.clip(max(0.22, pre_area_frac * 2.0), 0.22, 0.50))
        gx, gy, gw, gh = _expand_box_by_density(raw_mask01, x, y, w, h, max_frame_frac=grow_cap, debug=debug)
        if (gw * gh) / frame_area > 0.55:
            gx, gy, gw, gh = x, y, w, h
        static_frac, ring_ratio = _box_persistence(edge_freq, gx, gy, gw, gh)
        if static_frac < min_static_frac or ring_ratio < min_ring_ratio:
            continue
        if det_scale != 1.0:
            gx, gy, gw, gh = (int(round(v * det_scale)) for v in (gx, gy, gw, gh))
            box = WatermarkBox(gx, gy, gw, gh).clamped(true_w, true_h, pad=0)
        else:
            box = WatermarkBox(gx, gy, gw, gh).clamped(w_frame, h_frame, pad=0)
        if box.w > 0 and box.h > 0:
            box.conf = round(static_frac, 4)
            final_boxes.append(box)
    if debug:
        for b in final_boxes:
            print(f"  [watermark-detect] -> region x={b.x} y={b.y} w={b.w} h={b.h}")
    return final_boxes


#: Sensitivity ladder: (peak_floor, otsu_floor, density_floor, min_ring_ratio).
WM_LADDER = [
    (0.12, 0.10, 0.06, 1.4),
    (0.08, 0.07, 0.05, 1.6),
    (0.05, 0.045, 0.04, 2.0),
    (0.035, 0.03, 0.03, 2.4),
]
#: Search-area widening retries: (border_frac, min_area multiplier, max_area bonus).
WM_AREA_RETRIES = [
    (0.32, 1.0, 0.00),
    (0.45, 0.5, 0.10),
    (0.60, 0.25, 0.20),
]


def detect_watermark_auto(path: str, start_sec: float, end_sec: float, sample_count: int = 48,
                          min_area_frac: float = 0.0004, max_area_frac: float = 0.45,
                          density_floor: float = 0.06, max_regions: int = 4,
                          border_frac: float = 0.32, debug: bool = False) -> List[WatermarkBox]:
    """Dynamic watermark/logo/credit-text detection over the WHOLE usable window of the
    master: sensitivity ladder x search-area retries x (whole window + up to 4 segments).
    Short windows are sampled with more frames (a 3-10 second video still yields 56)."""
    usable = max(0.0, end_sec - start_sec)
    if usable <= 0:
        return []
    nseg = int(np.clip(usable // 12, 1, 4))
    _FRAME_CACHE.clear()
    count = 56 if usable < 20 else max(sample_count, 40) + (16 if nseg > 1 else 0)
    base = _read_gray_frames(path, start_sec, end_sec, count)
    all_f = base[0]
    windows = [(start_sec, end_sec, len(all_f), base)]
    if nseg > 1 and len(all_f) >= nseg * 6:
        n = len(all_f)
        for i in range(nseg):
            a, b = i * n // nseg, (i + 1) * n // nseg
            windows.append((start_sec + i * usable / nseg, start_sec + (i + 1) * usable / nseg, b - a,
                            (all_f[a:b], base[1], base[2], base[3])))

    for attempt, (bfrac, min_mult, max_bonus) in enumerate(WM_AREA_RETRIES, 1):
        use_border = max(border_frac, bfrac)
        for rung, (pk, tlo, dens, ring) in enumerate(WM_LADDER):
            found: List[WatermarkBox] = []
            for (ws, we, cnt, pre) in windows:
                try:
                    found += _detect_wm_pass(path, ws, we, sample_count=cnt, preloaded=pre,
                                             min_area_frac=min_area_frac * min_mult,
                                             max_area_frac=min(0.7, max_area_frac + max_bonus),
                                             density_floor=min(density_floor, dens),
                                             max_regions=max_regions, border_frac=use_border, debug=debug,
                                             peak_floor=pk, thresh_lo=tlo, min_ring_ratio=ring)
                except Exception as e:   # noqa: BLE001
                    if debug:
                        print(f"  [watermark-detect] window [{ws:.1f},{we:.1f}] failed: {e}")
            if found:
                if attempt > 1 or rung > 0:
                    print(f"  watermark found on retry: search attempt {attempt}/{len(WM_AREA_RETRIES)} "
                          f"(border {use_border:.2f}), sensitivity rung {rung + 1}/{len(WM_LADDER)}")
                return _merge_window_boxes(found, max_regions)
    return []


def _merge_window_boxes(boxes: List[WatermarkBox], max_regions: int) -> List[WatermarkBox]:
    items = [[b.x, b.y, b.w, b.h, b.conf] for b in boxes]
    changed = True
    while changed:
        changed = False
        out, used = [], [False] * len(items)
        for i in range(len(items)):
            if used[i]:
                continue
            cur = items[i]
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                gap = max(4, int(0.15 * min(max(cur[2], cur[3]), max(items[j][2], items[j][3]))))
                if _boxes_overlap_or_close(tuple(cur[:4]), tuple(items[j][:4]), gap):
                    x1, y1 = min(cur[0], items[j][0]), min(cur[1], items[j][1])
                    x2 = max(cur[0] + cur[2], items[j][0] + items[j][2])
                    y2 = max(cur[1] + cur[3], items[j][1] + items[j][3])
                    cur = [x1, y1, x2 - x1, y2 - y1, max(cur[4], items[j][4])]
                    used[j] = True
                    changed = True
            out.append(cur)
            used[i] = True
        items = out
    items.sort(key=lambda t: t[4], reverse=True)
    return [WatermarkBox(a, b, c, d, conf=e) for a, b, c, d, e in items[:max_regions]]


def watermark_from_pixels(spec: str) -> WatermarkBox:
    try:
        x, y, w, h = [int(v.strip()) for v in spec.split(",")]
    except Exception:
        sys.exit(f"--watermark-box must be 'x,y,w,h' in pixels, got: {spec!r}")
    return WatermarkBox(x, y, w, h)


def watermark_from_pct(spec: str, frame_w: int, frame_h: int) -> WatermarkBox:
    try:
        xp, yp, wp, hp = [float(v.strip()) for v in spec.split(",")]
    except Exception:
        sys.exit(f"--watermark-box-pct must be 'x,y,w,h' as fractions 0-1, got: {spec!r}")
    return watermark_box_from_fractions(xp, yp, wp, hp, frame_w, frame_h)


def save_watermark_preview(path: str, wms: List[WatermarkBox], out_path: str, at_sec: float = 2.0):
    """One frame with a red rectangle around every region that WOULD be removed."""
    frame_w, frame_h = ffprobe_dimensions(path)
    cmd = ["ffmpeg", "-y", "-ss", f"{at_sec:.2f}", "-i", path, "-frames:v", "1"]
    safe_boxes = [b for b in (wm.clamped(frame_w, frame_h) for wm in (wms or [])) if b.w > 0 and b.h > 0]
    if safe_boxes:
        cmd += ["-vf", ",".join(f"drawbox=x={b.x}:y={b.y}:w={b.w}:h={b.h}:color=red@0.9:thickness=4"
                                for b in safe_boxes)]
    cmd += [out_path]
    run(cmd)
    print(f"  -> preview saved: {out_path}  [{len(safe_boxes)} box(es)]")


# --------------------------------------------------------------------------- #
# Splitting the usable window into N-second parts
# --------------------------------------------------------------------------- #

def compute_parts(usable_start: float, usable_end: float, clip_sec: float,
                  keep_remainder: bool, two_part_fallback: bool = True,
                  two_part_min_sec: float = 10.0, equal_split: bool = True
                  ) -> List[Tuple[float, float]]:
    """List of (part_start, part_duration). Equal-split: the usable window is divided into the
    nearest whole number of roughly clip_sec-long parts so nothing is dropped as a remainder.
    A window shorter than clip_sec becomes two equal halves when it is long enough (the
    minimum scales down for small --clip-seconds, so 3 second clips still work)."""
    usable_duration = usable_end - usable_start
    if usable_duration <= 0:
        return []
    two_part_min_sec = min(two_part_min_sec, max(2.0, 2.0 * clip_sec * 0.5))

    if equal_split:
        if usable_duration < clip_sec:
            if two_part_fallback and usable_duration >= two_part_min_sec:
                half = usable_duration / 2.0
                return [(usable_start, half), (usable_start + half, half)]
            if usable_duration >= 2.0:
                return [(usable_start, usable_duration)]
            return []
        num_parts = max(1, round(usable_duration / clip_sec))
        part_len = usable_duration / num_parts
        return [(usable_start + i * part_len, part_len) for i in range(num_parts)]

    parts = []
    t = usable_start
    while t + clip_sec <= usable_end + 1e-6:
        parts.append((t, clip_sec))
        t += clip_sec
    if keep_remainder and usable_end - t > 2.0:
        parts.append((t, usable_end - t))
    return parts


# --------------------------------------------------------------------------- #
# Rendering (STAGE 3: clips are cut from the master)
# --------------------------------------------------------------------------- #

#: Removal-box growth steps tried during calibration (per box, on the master, no rendering).
GROW_STEPS = (1.0, 1.35, 1.8, 2.4)

#: Max extra calibration steps beyond 1.0x (CLI: --max-retry-renders). 0 = never grow the box.
MAX_RETRY_RENDERS = 4


def _removal_pad(wm: "WatermarkBox") -> Tuple[int, int]:
    """(pad_x, pad_y) around a detected box, per axis, so a long thin credit line does not
    balloon vertically into a huge box."""
    return max(6, int(round(0.08 * wm.w))), max(6, int(round(0.25 * wm.h)))


def _grow_box(wm: "WatermarkBox", factor: float) -> "WatermarkBox":
    if factor == 1.0:
        return wm
    nw, nh = int(round(wm.w * factor)), int(round(wm.h * factor))
    return WatermarkBox(wm.x - (nw - wm.w) // 2, wm.y - (nh - wm.h) // 2, nw, nh, conf=wm.conf)


def _padded(wm: "WatermarkBox", frame_w: int, frame_h: int, grow: float = 1.0) -> "WatermarkBox":
    px, py = _removal_pad(wm)
    base = _grow_box(wm, grow) if grow != 1.0 else wm
    return base.clamped(frame_w, frame_h, pad=px, pad_y=py)


def limit_watermark_boxes(wms: List["WatermarkBox"], frame_w: int, frame_h: int,
                          max_region_frac: float, max_total_frac: float,
                          edge_only: bool = True) -> List["WatermarkBox"]:
    """Safety net for AUTO-detected boxes: drop boxes that are too big, in the middle of the
    picture (unless strongly persistent), or that push the total removed area too high."""
    frame_area = float(frame_w * frame_h)
    edge_zone = 0.32 * min(frame_w, frame_h)
    kept: List["WatermarkBox"] = []
    total = 0.0
    for wm in wms:
        padded = _padded(wm, frame_w, frame_h)
        if padded.w <= 0 or padded.h <= 0:
            continue
        frac = (padded.w * padded.h) / frame_area
        cx, cy = wm.x + wm.w / 2.0, wm.y + wm.h / 2.0
        dist_edge = min(cx, cy, frame_w - cx, frame_h - cy)
        desc = f"x={wm.x} y={wm.y} w={wm.w} h={wm.h} ({frac:.1%} of frame, conf={wm.conf:.2f})"
        strong = wm.conf >= 0.08
        max_region_frac_eff = max(max_region_frac * 2.0, 0.25) if strong else max_region_frac
        if edge_only and not strong and dist_edge > edge_zone:
            _warn(f"Ignoring watermark candidate {desc}: middle of the picture and not strongly "
                  f"persistent (--watermark-allow-center to override)")
            continue
        if frac > max_region_frac_eff:
            _warn(f"Ignoring watermark candidate {desc}: larger than the {max_region_frac_eff:.0%} "
                  f"per-region cap (--watermark-max-area-pct, or --watermark-box-pct)")
            continue
        if total + frac > (max(max_total_frac * 1.5, 0.28) if strong else max_total_frac):
            _warn(f"Ignoring watermark candidate {desc}: total removal area would exceed "
                  f"{max_total_frac:.0%} of the frame (--watermark-total-area-pct)")
            continue
        kept.append(wm)
        total += frac
    return kept


def _rate_to_bits(rate: str) -> Optional[float]:
    m = re.match(r"^\s*([\d.]+)\s*([kKmM]?)\s*$", str(rate))
    if not m:
        return None
    mult = {"": 1.0, "k": 1e3, "m": 1e6}[m.group(2).lower()]
    return float(m.group(1)) * mult


def _build_render_cmd(input_path: str, output_path: str, start_sec: float, clip_sec: float,
                      filters: List[str], optimize: bool, crf: int, preset: str,
                      video_bitrate: str, audio_bitrate: str) -> List[str]:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start_sec:.3f}", "-i", input_path, "-t", f"{clip_sec:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-profile:v", "high"]
    if optimize:
        bits = _rate_to_bits(video_bitrate)
        cmd += ["-maxrate", video_bitrate,
                "-bufsize", f"{int(bits * 2 / 1000)}k" if bits else video_bitrate]
    cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2",
            "-movflags", "+faststart", output_path]
    return cmd


def _validate_output(path: str, expected_sec: float) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < 2048:
        raise RuntimeError(f"output missing or empty: {path}")
    dur = ffprobe_duration(path)
    if dur < max(0.5, 0.5 * expected_sec):
        raise RuntimeError(f"output too short ({dur:.1f}s, expected ~{expected_sec:.1f}s): {path}")


def render_clip(input_path: str, output_path: str, start_sec: float, clip_sec: float,
                watermarks: List[WatermarkBox], crf: Optional[int] = None,
                preset: Optional[str] = None, optimize: bool = True, max_dim: int = MAX_DIM,
                video_bitrate: str = VIDEO_BITRATE, audio_bitrate: str = AUDIO_BITRATE,
                encode_preset: str = DEFAULT_ENCODE_PRESET, grow=1.0) -> bool:
    """Renders one part from the master. Returns True if EVERY requested watermark region
    was removed (delogo or blur patch), False if something had to be left in.

    `grow` is a float or a per-box list of the calibrated removal-box growth factors.

    Fallback ladder (never silently skips the part, never silently keeps a logo if avoidable):
      1. delogo on all regions
      2. probe each region; keep delogo where ffmpeg accepts it, BLUR-PATCH the rest
      3. blur-patch every region
      4. no removal at all
      5. plain safe encode"""
    frame_w, frame_h = ffprobe_dimensions(input_path)
    if crf is None:
        crf = VIDEO_CRF if optimize else 18
    enc_preset = preset or encode_preset

    boxes: List[WatermarkBox] = []
    for i, watermark in enumerate(watermarks or []):
        g = grow[i] if isinstance(grow, (list, tuple)) else grow
        pb = _padded(watermark, frame_w, frame_h, g)
        if pb.w > 0 and pb.h > 0:
            boxes.append(pb)
        else:
            _warn("A watermark region was invalid after clamping to frame bounds -- skipped.")
    all_covered = len(boxes) == len(watermarks or [])

    scale_filters = _scale_filters(frame_w, frame_h, optimize, max_dim)
    timeout = max(600, int(clip_sec * 40))

    def _try(flt: List[str], preset_: str) -> Optional[Exception]:
        cmd = _build_render_cmd(input_path, output_path, start_sec, clip_sec, flt, optimize,
                                crf, preset_, video_bitrate, audio_bitrate)
        try:
            run(cmd, timeout=timeout)
            _validate_output(output_path, clip_sec)
            return None
        except Exception as e:   # noqa: BLE001
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass
            return e

    def _patches(bs: List[WatermarkBox], offset: int = 0) -> List[str]:
        return [b.as_blur_patch(offset + k) for k, b in enumerate(bs)]

    last_err: Optional[Exception] = None
    if boxes:
        last_err = _try([b.as_ffmpeg_delogo() for b in boxes] + scale_filters, enc_preset)
        if last_err is None:
            return all_covered

        # which regions does delogo accept? the rest get a blur patch instead of being dropped
        good, bad = [], []
        for b in boxes:
            probe_out = output_path + ".probe.mp4"
            cmd = _build_render_cmd(input_path, probe_out, start_sec, min(1.0, clip_sec),
                                    [b.as_ffmpeg_delogo()] + scale_filters, optimize, 30, "ultrafast",
                                    video_bitrate, audio_bitrate)
            try:
                run(cmd, timeout=120)
                good.append(b)
            except Exception:   # noqa: BLE001
                bad.append(b)
            finally:
                if os.path.exists(probe_out):
                    os.remove(probe_out)
        if good and bad:
            flt = [b.as_ffmpeg_delogo() for b in good] + _patches(bad) + scale_filters
            err = _try(flt, enc_preset)
            if err is None:
                _warn(f"{len(bad)} region(s) were not accepted by delogo; covered with a blur patch instead.")
                return all_covered
            last_err = err
        err = _try(_patches(boxes) + scale_filters, enc_preset)
        if err is None:
            _warn("delogo was rejected by ffmpeg; every watermark region was covered with a blur patch.")
            return all_covered
        last_err = err
        _warn("watermark removal failed -- rendering this part WITHOUT removal.")
        err = _try(scale_filters, enc_preset)
        if err is None:
            return False
        last_err = err
    else:
        last_err = _try(scale_filters, enc_preset)
        if last_err is None:
            return all_covered

    _warn("normal render failed -- trying a plain safe encode.")
    err = _try(scale_filters, "veryfast")
    if err is None:
        return False
    raise err if err else last_err  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Watermark: ONE calibration per video on the master (replaces per-part re-renders)
# --------------------------------------------------------------------------- #

def _edge_freq_map(path: str, t0: float, t1: float, count: int = 40):
    """Per-pixel frequency of Canny edges across frames of [t0, t1] -> (map, true_w, true_h)."""
    frames, tw, th, _scale = _read_gray_frames(path, t0, max(t0 + 0.5, t1), count)
    if len(frames) < 6:
        return None
    med = float(np.median(np.stack([f[::8, ::8] for f in frames])))
    c_lo = int(np.clip(0.66 * med, 25, 70))
    c_hi = int(np.clip(c_lo * 2.4, 70, 180))
    acc = sum(cv2.Canny(f, c_lo, c_hi).astype(np.float32) for f in frames) / (255.0 * len(frames))
    return acc, tw, th


def _band_persistence(acc: np.ndarray, tw: int, th: int, inner: "WatermarkBox") -> float:
    """Fraction of pixels in a band AROUND `inner` (the region that will be removed) that still
    show persistent edges. High => the overlay continues beyond the removal box (it will leak)."""
    H, W = acc.shape
    sx, sy = W / float(tw), H / float(th)
    bw = max(6, int(0.3 * max(inner.w, inner.h)))
    ox0, oy0 = max(0, int((inner.x - bw) * sx)), max(0, int((inner.y - bw) * sy))
    ox1, oy1 = min(W, int((inner.x + inner.w + bw) * sx)), min(H, int((inner.y + inner.h + bw) * sy))
    ix0, iy0 = max(0, int(inner.x * sx)), max(0, int(inner.y * sy))
    ix1, iy1 = min(W, int((inner.x + inner.w) * sx)), min(H, int((inner.y + inner.h) * sy))
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    region = acc[oy0:oy1, ox0:ox1] >= 0.4
    mask = np.ones(region.shape, dtype=bool)
    mask[max(0, iy0 - oy0):max(0, iy1 - oy0), max(0, ix0 - ox0):max(0, ix1 - ox0)] = False
    n = int(mask.sum())
    return float(region[mask].sum()) / n if n else 0.0


def calibrate_removal_boxes(path: str, wms: List["WatermarkBox"], frame_w: int, frame_h: int,
                            t0: float, t1: float, max_total_frac: float
                            ) -> Tuple[List[float], bool]:
    """Pick, per detected box, the growth factor whose removal region leaves the least
    persistent overlay just outside it. Works on the master's frames only, so it costs no
    encode. The smallest factor within 15% of the best result wins, so a static background
    (which also shows persistent edges) cannot make the box grow without limit.
    Returns (grow_per_box, any_box_still_leaking)."""
    grows = [1.0] * len(wms)
    if not wms:
        return grows, False
    emap = _edge_freq_map(path, t0, t1)
    if emap is None:
        return grows, False
    acc, tw, th = emap
    frame_area = float(frame_w * frame_h)
    steps = GROW_STEPS[:1 + max(0, MAX_RETRY_RENDERS)]
    leaking = False
    for i, wm in enumerate(wms):
        results = []
        for g in steps:
            p = _padded(wm, frame_w, frame_h, g)
            if p.w <= 0 or p.h <= 0:
                break
            if g > 1.0 and (p.w * p.h) / frame_area > max(max_total_frac * 1.5, 0.30):
                break
            results.append((g, _band_persistence(acc, tw, th, p)))
        if not results:
            continue
        best_leak = min(r for _g, r in results)
        pick = next(g for g, r in results if r <= best_leak * 1.15 + 1e-6)
        grows[i] = pick
        thr = max(0.04, 0.3 * wm.conf)
        chosen = dict(results)[pick]
        if chosen > thr * 1.5:
            leaking = True
    return grows, leaking


# --------------------------------------------------------------------------- #
# Batch-consensus watermark (fallback for videos where own detection found nothing)
# --------------------------------------------------------------------------- #

def _median_hp_ratio(med_img: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[float, float]:
    H, W = med_img.shape
    hp = np.abs(med_img - cv2.GaussianBlur(med_img, (0, 0), 3))
    inner_sum = float(hp[y:y + h, x:x + w].sum())
    inner = inner_sum / max(1, w * h)
    pad = max(6, int(0.5 * max(w, h)))
    outer = hp[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    n_ring = outer.size - w * h
    ring = (float(outer.sum()) - inner_sum) / n_ring if n_ring > 0 else 0.0
    return inner, inner / (ring + 1e-3)


def verify_boxes_on_video(path: str, wms: List[WatermarkBox], start_sec: float, end_sec: float
                          ) -> List[WatermarkBox]:
    """Keep only candidate boxes that really contain a persistent static overlay in THIS video."""
    if not wms:
        return []
    frames, tw, th, scale = _read_gray_frames(path, start_sec, max(start_sec + 0.5, end_sec), 28)
    if len(frames) < 6:
        return []
    med = float(np.median(np.stack([f[::8, ::8] for f in frames])))
    c_lo = int(np.clip(0.66 * med, 25, 70))
    c_hi = int(np.clip(c_lo * 2.4, 70, 180))
    acc = sum(cv2.Canny(f, c_lo, c_hi).astype(np.float32) for f in frames) / (255.0 * len(frames))
    H, W = acc.shape
    med_img = np.median(np.stack(frames), axis=0).astype(np.float32)
    ok_boxes = []
    for b in wms:
        x, y = int(b.x / scale), int(b.y / scale)
        w, h = max(2, int(b.w / scale)), max(2, int(b.h / scale))
        x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
        w, h = min(w, W - x), min(h, H - y)
        static_frac, ring_ratio = _box_persistence(acc, x, y, w, h)
        if static_frac >= 0.008 and ring_ratio >= 1.25:
            b.conf = round(static_frac, 4)
            ok_boxes.append(b)
            continue
        hp_inner, hp_ratio = _median_hp_ratio(med_img, x, y, w, h)
        if hp_inner >= 2.0 and hp_ratio >= 1.35:
            b.conf = max(b.conf, 0.08)
            ok_boxes.append(b)
    return ok_boxes


def consensus_boxes(fraction_sets, min_votes: int = 2, min_share: float = 0.3):
    flat = [(i, f) for i, fs in enumerate(fraction_sets) for f in fs]
    if not flat:
        return []

    def iou(a, b):
        ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
        iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
        ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
        inter = iw * ih
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union > 0 else 0.0

    clusters = []
    for item in flat:
        for c in clusters:
            if iou(item[1], c[0][1]) > 0.35:
                c.append(item)
                break
        else:
            clusters.append([item])
    n_with = sum(1 for fs in fraction_sets if fs)
    need = max(min_votes, int(np.ceil(min_share * n_with)))
    out = []
    for c in clusters:
        voters = {i for i, _ in c}
        if len(voters) >= need:
            xs = [f[0] for _, f in c]
            ys = [f[1] for _, f in c]
            x2 = [f[0] + f[2] for _, f in c]
            y2 = [f[1] + f[3] for _, f in c]
            x0, y0 = float(np.median(xs)), float(np.median(ys))
            out.append((x0, y0, float(np.median(x2)) - x0, float(np.median(y2)) - y0))
    return out


# --------------------------------------------------------------------------- #
# Pipeline for one video (STAGE 2 = plan on the master, STAGE 3 = execute)
# --------------------------------------------------------------------------- #

def process_single(path: str, master_path: Optional[str], out_path_template: str, clip_sec: float,
                   watermark_mode: str, dry_run: bool, single_clip: bool, keep_remainder: bool,
                   shared_intro_end: Optional[float], shared_outro_len: Optional[float],
                   intro_override: Optional[float], outro_override: Optional[float],
                   watermark_box_spec: Optional[str], watermark_box_pct_spec: Optional[str],
                   debug_preview: bool, debug_detect: bool,
                   intro_max_search: float = DEFAULT_INTRO_MIN_SCAN_SEC,
                   outro_max_search: float = DEFAULT_OUTRO_MAX_SEARCH_SEC,
                   outro_safety_margin: float = DEFAULT_OUTRO_SAFETY_MARGIN_SEC,
                   two_part_fallback: bool = True, two_part_min_sec: float = 10.0,
                   watermark_max_area_frac: float = 0.45,
                   optimize: bool = True, max_dim: int = MAX_DIM,
                   video_bitrate: str = VIDEO_BITRATE, audio_bitrate: str = AUDIO_BITRATE,
                   encode_preset: str = DEFAULT_ENCODE_PRESET, crf: Optional[int] = None,
                   wm_total_frac: float = 0.10, wm_edge_only: bool = True,
                   plan_only: bool = False, master_notes: Optional[List[str]] = None):
    """Plans (and, unless plan_only, renders) one video. ALL analysis reads `master_path`
    (the optimized whole video); the original is only used if no master could be built.
    Returns the number of output files written, or the plan dict if plan_only."""
    CURRENT_NOTES.clear()
    CURRENT_NOTES.extend(master_notes or [])
    orig_name = os.path.basename(path)
    src = master_path or path           # <- everything below analyses and cuts THIS file
    duration = ffprobe_duration(src)
    print(f"[{orig_name}]" + ("  (analysing optimized master)" if master_path else "  (analysing original)"))

    # A trim must always leave something worth keeping; the limit scales down for short videos.
    min_keep = max(1.0, min(5.0, 0.15 * duration))

    # The outro safety margin scales with the video (2s on a 5 minute video would eat a third of a 10s one).
    eff_margin = min(outro_safety_margin, max(0.3, 0.06 * duration))

    def decide_trims(use_shared: bool):
        """(intro, outro_start, outro_method, outro_conf) for this video. A detector crash must
        never skip the video: it falls back to 'nothing to trim'."""
        try:
            if intro_override is not None:
                intro_ = BoundaryResult(intro_override, "manual_override", "high")
            elif use_shared and shared_intro_end is not None:
                intro_ = BoundaryResult(shared_intro_end, "batch_common_prefix", "high")
            else:
                intro_ = detect_intro_with_retries(src, max_search_sec=intro_max_search, debug=debug_detect)
        except Exception as e:   # noqa: BLE001
            _warn(f"intro detection crashed ({str(e).strip()[-120:]}); assuming no intro")
            intro_ = BoundaryResult(0.0, "detect_error", "none")
        if intro_.time_sec > duration - min_keep:
            intro_ = BoundaryResult(0.0, intro_.method + "_rejected_too_long", "none")
        try:
            if outro_override is not None:
                o_start, o_method, o_conf = outro_override, "manual_override", "high"
            elif use_shared and shared_outro_len is not None:
                o_start, o_method, o_conf = duration - shared_outro_len, "batch_common_suffix", "high"
            else:
                o = detect_outro_with_retries(src, max_search_sec=outro_max_search,
                                              safety_margin=eff_margin, debug=debug_detect)
                o_start, o_method, o_conf = o.time_sec, o.method, o.confidence
        except Exception as e:   # noqa: BLE001
            _warn(f"outro detection crashed ({str(e).strip()[-120:]}); assuming no outro")
            o_start, o_method, o_conf = duration, "detect_error", "none"
        if o_start < intro_.time_sec + min_keep:
            o_start, o_method, o_conf = duration, o_method + "_rejected_too_short", "none"
        return intro_, min(o_start, duration), o_method, o_conf

    intro, outro_start, outro_method, outro_conf = decide_trims(True)
    if (shared_intro_end is not None or shared_outro_len is not None) and outro_start - intro.time_sec < 2.0:
        _warn("batch-shared intro/outro would leave under 2s of this video -- using this video's own "
              "intro/outro detection instead")
        intro, outro_start, outro_method, outro_conf = decide_trims(False)
    usable_start, usable_end = intro.time_sec, outro_start

    # --- ONE refinement check of the cut points (on the master, not per part) ---
    check_intro = intro.time_sec > 0 and intro.confidence != "none" and intro.method != "manual_override"
    check_outro = outro_start < duration - 0.3 and outro_conf != "none" and outro_method != "manual_override"
    if check_intro or check_outro:
        try:
            lead, tail = refine_boundaries_on_master(src, usable_start, usable_end,
                                                     check_intro, check_outro, debug=debug_detect)
            if lead or tail:
                usable_start += lead
                usable_end -= tail
                _warn(f"cut-point check: leftover intro/outro material found -- trimmed another "
                      f"{lead:.2f}s at the start and {tail:.2f}s at the end (once, for the whole video).")
        except Exception as e:   # noqa: BLE001
            _warn(f"cut-point check skipped ({str(e).strip()[-100:]})")

    # --- watermark / logo / credit text: detected ONCE over the whole usable window ---
    frame_w, frame_h = ffprobe_dimensions(src)
    wm_notes = ""
    if watermark_box_spec is not None:
        wms = [watermark_from_pixels(watermark_box_spec)]
        wm_source = "manual_pixels"
    elif watermark_box_pct_spec is not None:
        wms = [watermark_from_pct(watermark_box_pct_spec, frame_w, frame_h)]
        wm_source = "manual_pct"
    elif watermark_mode == "auto":
        try:
            wms = detect_watermark_auto(src, start_sec=usable_start, end_sec=usable_end,
                                        max_area_frac=watermark_max_area_frac, debug=debug_detect)
        except Exception as e:   # noqa: BLE001
            _warn(f"watermark detection crashed ({str(e).strip()[-120:]})")
            wms = []
        wm_source = "auto_detect"
        if wms:
            wms = limit_watermark_boxes(wms, frame_w, frame_h,
                                        max_region_frac=watermark_max_area_frac,
                                        max_total_frac=wm_total_frac, edge_only=wm_edge_only)
        if not wms:
            wm_notes = "no watermark detected (faint/absent?). Try --debug-detect --debug-preview or --watermark-box-pct"
            print(f"  note: {wm_notes}")
    elif watermark_mode == "manual":
        manual_box = detect_watermark_manual(src)
        wms = [manual_box] if manual_box is not None else []
        wm_source = "manual_select"
    else:
        wms = []
        wm_source = "none"

    # --- calibrate the removal boxes once (master frames only; no encode) ---
    grows = [1.0] * len(wms)
    leaking = False
    if wms and wm_source in ("auto_detect", "manual_select"):
        try:
            grows, leaking = calibrate_removal_boxes(src, wms, frame_w, frame_h, usable_start, usable_end,
                                                     wm_total_frac)
        except Exception as e:   # noqa: BLE001
            _warn(f"removal-box calibration skipped ({str(e).strip()[-100:]})")

    # --- parts ---
    if single_clip:
        parts = [(usable_start, min(clip_sec, max(0.0, usable_end - usable_start)))]
    else:
        parts = compute_parts(usable_start, usable_end, clip_sec, keep_remainder,
                              two_part_fallback=two_part_fallback, two_part_min_sec=two_part_min_sec)

    min_part = min(2.0, 0.5 * clip_sec)
    if (not parts or parts[0][1] < min_part) and duration >= 2.0:
        usable = max(0.0, usable_end - usable_start)
        if usable >= 2.0:
            parts = [(usable_start, min(usable, clip_sec) if single_clip else usable)]
            _warn(f"usable window is only {usable:.1f}s -- kept as one short part instead of skipping")
        else:
            _warn(f"intro/outro trimming left {usable:.1f}s -- ignoring detected trims and using the whole video")
            usable_start, usable_end = 0.0, duration
            if single_clip:
                parts = [(0.0, min(clip_sec, duration))]
            else:
                parts = compute_parts(0.0, duration, clip_sec, True, two_part_fallback=True,
                                      two_part_min_sec=min(two_part_min_sec, 4.0)) or [(0.0, duration)]

    print(f"  duration         : {duration:.2f}s")
    print(f"  intro end        : {usable_start:.2f}s  (detected {intro.time_sec:.2f}s via {intro.method}, {intro.confidence})")
    print(f"  outro start      : {usable_end:.2f}s  (detected {outro_start:.2f}s via {outro_method}, {outro_conf})")
    print(f"  usable window    : {usable_start:.2f}s -> {usable_end:.2f}s  ({max(0.0, usable_end-usable_start):.2f}s usable)")
    if wms:
        desc = "; ".join(f"x={b.x} y={b.y} w={b.w} h={b.h} conf={b.conf:.2f} grow={g:.2f}x"
                         for b, g in zip(wms, grows))
        print(f"  watermark region(s): {len(wms)} ({wm_source}) -- {desc}")
        if leaking:
            _warn("overlay seems to continue beyond the removal box even at the largest allowed size -- "
                  "check with --debug-preview or give --watermark-box-pct")
    else:
        print("  watermark region(s): none")
    print(f"  parts to produce : {len(parts)}")
    for i, (pstart, pdur) in enumerate(parts, 1):
        print(f"    part {i:02d}: {pstart:.2f}s -> {pstart+pdur:.2f}s  ({pdur:.2f}s)")

    if debug_preview:
        base_dir = os.path.dirname(out_path_template) or "."
        base_name = os.path.splitext(os.path.basename(out_path_template))[0]
        os.makedirs(base_dir, exist_ok=True)
        preview_at = min(usable_start + (usable_end - usable_start) / 2.0, max(0.0, usable_end - 0.5))
        save_watermark_preview(src, wms, os.path.join(base_dir, f"{base_name}_wm_preview.png"), at_sec=preview_at)

    plan = dict(path=src, orig_name=orig_name, master_path=master_path, out_path_template=out_path_template,
                parts=parts, wms=wms, grows=grows, wm_leaking=leaking, wm_source=wm_source,
                frame_w=frame_w, frame_h=frame_h, crf=crf, optimize=optimize, max_dim=max_dim,
                video_bitrate=video_bitrate, audio_bitrate=audio_bitrate, encode_preset=encode_preset,
                notes=list(CURRENT_NOTES), dry_run=dry_run, duration=duration,
                usable=(usable_start, usable_end),
                intro_info=dict(end=round(usable_start, 2), method=intro.method, confidence=intro.confidence),
                outro_info=dict(start=round(usable_end, 2), method=outro_method, confidence=outro_conf))
    if plan_only:
        return plan
    return execute_plan(plan)


def execute_plan(plan: dict) -> int:
    """Cuts every part from the master with the SAME trim and the SAME calibrated watermark
    removal. Nothing is re-detected or re-checked per part, so a 3 second part is handled
    exactly like a 60 second one."""
    CURRENT_NOTES[:] = plan["notes"]
    src, parts, wms = plan["path"], plan["parts"], plan["wms"]
    out_path_template = plan["out_path_template"]
    entry = dict(name=plan.get("orig_name") or os.path.basename(src), parts_planned=len(parts), written=0,
                 wm_source=plan["wm_source"], wm_regions=len(wms), notes=CURRENT_NOTES,
                 wm_boxes=[dict(x=w.x, y=w.y, w=w.w, h=w.h, conf=w.conf, grow=g)
                           for w, g in zip(wms, plan.get("grows") or [1.0] * len(wms))],
                 intro=plan.get("intro_info"), outro=plan.get("outro_info"),
                 outputs=[], wm_unresolved=bool(plan.get("wm_leaking")), wm_not_found=False,
                 retry_renders=0, used_master=bool(plan.get("master_path")))

    def _finish(n: int) -> int:
        entry["written"] = n
        if plan["wm_source"] == "auto_detect" and not wms:
            entry["wm_not_found"] = True
        entry["notes"] = list(CURRENT_NOTES)
        REPORT.append(entry)
        mp = plan.get("master_path")
        if mp and os.path.exists(mp):
            try:
                os.remove(mp)            # free disk as soon as this video is done
            except OSError:
                pass
        return n

    if plan["dry_run"]:
        return _finish(0)

    base_dir = os.path.dirname(out_path_template) or "."
    base_name = os.path.splitext(os.path.basename(out_path_template))[0]
    ext = os.path.splitext(out_path_template)[1] or ".mp4"
    os.makedirs(base_dir, exist_ok=True)

    if len(parts) == 0:
        _warn("No parts to render (video shorter than 2s or unreadable duration).")
        return _finish(0)

    written = 0
    removal_gaps = 0
    kw = dict(crf=plan["crf"], optimize=plan["optimize"], max_dim=plan["max_dim"],
              video_bitrate=plan["video_bitrate"], audio_bitrate=plan["audio_bitrate"],
              encode_preset=plan["encode_preset"], grow=plan.get("grows") or 1.0)
    for i, (pstart, pdur) in enumerate(parts, 1):
        if pdur <= 0:
            continue
        out_path = (os.path.join(base_dir, base_name + ext) if len(parts) == 1
                    else os.path.join(base_dir, f"{base_name}_part{i:02d}{ext}"))
        try:
            applied = render_clip(src, out_path, pstart, pdur, wms, **kw)
            if wms and not applied:
                removal_gaps += 1
        except Exception as e:   # noqa: BLE001 - keep the other parts of this video
            _warn(f"part {i:02d} failed and was skipped: {str(e).strip()[-400:]}")
            continue
        written += 1
        entry["outputs"].append(os.path.basename(out_path))
        print(f"  -> wrote {out_path}")
    if removal_gaps:
        entry["wm_unresolved"] = True
        _warn(f"{removal_gaps} part(s) could not get full watermark removal (see messages above).")
    return _finish(written)


def write_report_json(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(REPORT, f, indent=2, default=str)
        print(f"Report JSON written to {path}")
    except OSError as e:
        print(f"  !! could not write report JSON: {e}")


def print_report() -> None:
    if not REPORT:
        return
    print("\n" + "=" * 78)
    print("REPORT (per video)")
    print("=" * 78)
    for e in REPORT:
        status = "OK " if e["written"] > 0 else "NONE"
        print(f"[{status}] {e['name']}: {e['written']}/{e['parts_planned']} part(s), "
              f"watermark={e['wm_regions']} region(s) via {e['wm_source']}")
        for n in e["notes"]:
            print(f"        - {n}")
    skipped = [e for e in REPORT if e["written"] == 0]
    nowm = [e for e in REPORT if e["wm_regions"] == 0 and e["wm_source"] == "auto_detect"]
    print("-" * 78)
    print(f"{len(REPORT) - len(skipped)} produced output, {len(skipped)} produced none, "
          f"{len(nowm)} had no watermark found.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def detect_watermark_manual(path: str) -> Optional[WatermarkBox]:
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Could not read a frame for manual selection.")
        return None
    print("Drag a box around the logo/watermark, then press ENTER or SPACE. 'c' to cancel.")
    box = cv2.selectROI("Select watermark - ENTER to confirm, C to cancel", frame,
                        showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = box
    if w == 0 or h == 0:
        return None
    return WatermarkBox(int(x), int(y), int(w), int(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="Single input video path")
    ap.add_argument("--output", help="Output path/template for single-video mode (parts get _partNN suffix)")
    ap.add_argument("--batch", help="Folder of input videos for batch mode")
    ap.add_argument("--outdir", help="Output folder for batch mode")
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--clip-seconds", type=float, default=60.0, help="Length of each part (default 60s; 3-10s work too)")
    ap.add_argument("--single-clip", action="store_true", help="Produce only ONE clip per video.")
    ap.add_argument("--keep-remainder", action="store_true", help="(legacy split only) keep a final short leftover part.")
    ap.add_argument("--two-part-fallback", dest="two_part_fallback", action="store_true", default=True)
    ap.add_argument("--no-two-part-fallback", dest="two_part_fallback", action="store_false")
    ap.add_argument("--two-part-min-sec", type=float, default=10.0)
    ap.add_argument("--watermark", choices=["auto", "manual", "none"], default="auto")
    ap.add_argument("--shared-watermark-from-first", action="store_true",
                    help="Batch mode: if no position recurs across the batch, reuse the first video's "
                         "detected position as the (verified) fallback.")
    ap.add_argument("--watermark-box", default=None, help="Fixed pixel box 'x,y,w,h' on the (master) frame.")
    ap.add_argument("--watermark-box-pct", default=None, help="Box as fractions 'x,y,w,h' (0-1), rescaled per video.")
    ap.add_argument("--watermark-max-area-pct", type=float, default=6.0)
    ap.add_argument("--watermark-total-area-pct", type=float, default=10.0)
    ap.add_argument("--watermark-allow-center", dest="wm_edge_only", action="store_false", default=True)
    ap.add_argument("--crf", type=int, default=None, help=f"Output quality (x264 CRF). Default {VIDEO_CRF}.")
    ap.add_argument("--intro-max-search", type=float, default=DEFAULT_INTRO_MIN_SCAN_SEC)
    ap.add_argument("--outro-max-search", type=float, default=DEFAULT_OUTRO_MAX_SEARCH_SEC)
    ap.add_argument("--outro-safety-margin", type=float, default=DEFAULT_OUTRO_SAFETY_MARGIN_SEC)
    ap.add_argument("--max-dim", type=int, default=MAX_DIM, help=f"Longest side in px (default {MAX_DIM}).")
    ap.add_argument("--video-bitrate", default=VIDEO_BITRATE, help="Bitrate ceiling of the final clips.")
    ap.add_argument("--audio-bitrate", default=AUDIO_BITRATE)
    ap.add_argument("--encode-preset", default=DEFAULT_ENCODE_PRESET)
    ap.add_argument("--no-optimize", dest="optimize", action="store_false", default=True)
    ap.add_argument("--no-master", dest="use_master", action="store_false", default=True,
                    help="Skip stage 1: analyse and cut the original file directly (debug / fallback).")
    ap.add_argument("--master-crf", type=int, default=MASTER_CRF,
                    help=f"Quality of the stage-1 master (default {MASTER_CRF}; lower = bigger, safer for the final encode).")
    ap.add_argument("--debug-preview", action="store_true")
    ap.add_argument("--report-json", default=os.environ.get("EDITOR_REPORT_JSON"))
    ap.add_argument("--time-budget-min", type=float, default=float(os.environ.get("EDITOR_TIME_BUDGET_MIN") or 0))
    ap.add_argument("--max-retry-renders", type=int, default=MAX_RETRY_RENDERS,
                    help="Max extra growth steps tried per watermark box during calibration (0 = never grow the box).")
    ap.add_argument("--debug-detect", action="store_true")
    ap.add_argument("--intro-sec", type=float, default=None)
    ap.add_argument("--outro-sec", type=float, default=None)
    ap.add_argument("--no-outro-detect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    globals()["MAX_RETRY_RENDERS"] = max(0, args.max_retry_renders)

    common = dict(
        watermark_box_spec=args.watermark_box, watermark_box_pct_spec=args.watermark_box_pct,
        debug_preview=args.debug_preview, debug_detect=args.debug_detect,
        intro_max_search=args.intro_max_search, outro_max_search=args.outro_max_search,
        outro_safety_margin=args.outro_safety_margin,
        two_part_fallback=args.two_part_fallback, two_part_min_sec=args.two_part_min_sec,
        watermark_max_area_frac=args.watermark_max_area_pct / 100.0,
        optimize=args.optimize, max_dim=args.max_dim,
        video_bitrate=args.video_bitrate, audio_bitrate=args.audio_bitrate,
        encode_preset=args.encode_preset, crf=args.crf,
        wm_total_frac=args.watermark_total_area_pct / 100.0, wm_edge_only=args.wm_edge_only)

    master_dir = tempfile.mkdtemp(prefix="vae_master_")
    try:
        if args.mode == "single":
            if not args.input:
                sys.exit("--input is required in single mode")
            out_template = args.output or (os.path.splitext(args.input)[0] + "_out.mp4")
            outro_override = args.outro_sec if not args.no_outro_detect else 10**9
            mpath, mnotes = prepare_master(args.input, master_dir, 0, args.use_master, args.optimize,
                                           args.max_dim, args.audio_bitrate, args.master_crf)
            process_single(args.input, mpath if mpath != args.input else None, out_template,
                           args.clip_seconds, args.watermark, args.dry_run, args.single_clip,
                           args.keep_remainder, shared_intro_end=None, shared_outro_len=None,
                           intro_override=args.intro_sec, outro_override=outro_override,
                           master_notes=mnotes, **common)
            print_report()
            write_report_json(args.report_json)
            return

        # ------------------------------ batch ------------------------------ #
        if not args.batch:
            sys.exit("--batch <folder> is required in batch mode")
        outdir = args.outdir or (args.batch.rstrip("/\\") + "_out")
        os.makedirs(outdir, exist_ok=True)

        vid_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".ts", ".flv", ".wmv",
                    ".mpg", ".mpeg", ".3gp", ".mts", ".m2ts", ".ogv"}
        paths = sorted(os.path.join(args.batch, f) for f in os.listdir(args.batch)
                       if os.path.isfile(os.path.join(args.batch, f))
                       and os.path.splitext(f)[1].lower() in vid_exts)
        if not paths:
            sys.exit(f"No videos found in {args.batch}")
        print(f"Found {len(paths)} videos.")

        readable = [p for p in paths if is_readable_video(p)]
        for bad in sorted(set(paths) - set(readable)):
            print(f"  !! Skipping unreadable/corrupt file: {os.path.basename(bad)}")
            REPORT.append(dict(name=os.path.basename(bad), parts_planned=0, written=0, wm_source="n/a",
                               wm_regions=0, notes=["unreadable/corrupt (ffprobe found no video stream or duration)"]))
        paths = readable
        if not paths:
            sys.exit("No readable videos in the batch.")

        t_start = time.time()
        budget_s = args.time_budget_min * 60.0 if args.time_budget_min and args.time_budget_min > 0 else 0.0
        n_failed = 0

        def _defer(p_, why):
            REPORT.append(dict(name=os.path.basename(p_), parts_planned=0, written=0, wm_source="n/a",
                               wm_regions=0, deferred=True, outputs=[], notes=[why]))

        # STAGE 1: optimized master of every whole video
        masters = {}   # original path -> (analysis path, notes)
        todo = []
        for idx, p in enumerate(paths):
            if budget_s and time.time() - t_start > budget_s:
                _defer(p, "deferred: editor time budget reached before this video was optimized (retry next run)")
                n_failed += 1
                continue
            mp, notes = prepare_master(p, master_dir, idx, args.use_master, args.optimize,
                                       args.max_dim, args.audio_bitrate, args.master_crf)
            masters[p] = (mp, notes)
            todo.append(p)
        paths = todo
        analysis_paths = [masters[p][0] for p in paths]

        # batch-wide shared intro/outro (measured on the masters)
        shared_intro_end = args.intro_sec
        shared_outro_len = None
        if shared_intro_end is None and len(paths) >= 2:
            print("Detecting shared intro across batch...")
            try:
                r = detect_intro_batch(analysis_paths, scan_seconds=args.intro_max_search)
                print(f"  -> {r.time_sec:.2f}s (confidence={r.confidence})")
                if r.confidence != "none":
                    shared_intro_end = r.time_sec
            except Exception as e:   # noqa: BLE001
                print(f"  !! shared-intro detection failed ({e}); using per-video detection")
        if args.outro_sec is None and not args.no_outro_detect and len(paths) >= 2:
            print("Detecting shared outro across batch...")
            try:
                outro_len = detect_outro_batch(analysis_paths, safety_margin=args.outro_safety_margin)
            except Exception as e:   # noqa: BLE001
                print(f"  !! shared-outro detection failed ({e}); using per-video detection")
                outro_len = None
            if outro_len:
                print(f"  -> {outro_len:.2f}s shared outro length, including safety margin")
                shared_outro_len = outro_len
            else:
                print("  -> no confident shared outro found; falling back to per-video detection")

        per_video_outro_override = None
        if args.outro_sec is not None:
            per_video_outro_override = args.outro_sec
        elif args.no_outro_detect:
            per_video_outro_override = 10**9

        import traceback
        used_bases = set()

        def _out_template(p):
            base = os.path.splitext(os.path.basename(p))[0]
            if base.lower() in used_bases:
                base = base + "_" + os.path.splitext(p)[1].lstrip(".").lower()
            used_bases.add(base.lower())
            return os.path.join(outdir, base + ".mp4")

        # STAGE 2: analyse every master once (intro / outro / watermark / parts), nothing rendered yet
        plans = []
        for p in paths:
            if budget_s and time.time() - t_start > budget_s:
                _defer(p, "deferred: editor time budget reached before this video was analysed (retry next run)")
                n_failed += 1
                mp = masters[p][0]
                if mp != p and os.path.exists(mp):
                    os.remove(mp)
                continue
            mp, mnotes = masters[p]
            try:
                plan = process_single(
                    p, mp if mp != p else None, _out_template(p), args.clip_seconds, args.watermark,
                    args.dry_run, args.single_clip, args.keep_remainder,
                    shared_intro_end=shared_intro_end, shared_outro_len=shared_outro_len,
                    intro_override=None if shared_intro_end is not None else args.intro_sec,
                    outro_override=per_video_outro_override, plan_only=True, master_notes=mnotes, **common)
                plans.append(plan)
            except Exception as e:   # noqa: BLE001
                print(f"  !! ERROR planning {os.path.basename(p)} -- skipping this video.")
                traceback.print_exc()
                REPORT.append(dict(name=os.path.basename(p), parts_planned=0, written=0, wm_source="n/a",
                                   wm_regions=0, notes=[f"crashed while planning: {str(e).strip()[-200:]}"]))
                n_failed += 1
                if mp != p and os.path.exists(mp):
                    os.remove(mp)

        # batch consensus: only a VERIFIED fallback for videos whose own detection found nothing
        auto_mode = (args.watermark == "auto" and args.watermark_box is None and args.watermark_box_pct is None)
        if auto_mode and len(plans) >= 2:
            frac_sets = [[w.to_fractions(pl["frame_w"], pl["frame_h"]) for w in pl["wms"]]
                         for pl in plans if pl["wm_source"] == "auto_detect"]
            cons = consensus_boxes(frac_sets)
            if not cons and args.shared_watermark_from_first:
                cons = next((fs for fs in frac_sets if fs), None) or []
            if cons:
                print(f"\nBatch-consensus watermark position(s) (fractions x,y,w,h): "
                      f"{[tuple(round(v, 3) for v in c) for c in cons]}")
                for pl in plans:
                    if pl["wm_source"] == "auto_detect" and not pl["wms"]:
                        cand = [watermark_box_from_fractions(xp, yp, wp, hp, pl["frame_w"], pl["frame_h"])
                                for (xp, yp, wp, hp) in cons]
                        ok = verify_boxes_on_video(pl["path"], cand, *pl["usable"])
                        name = pl["orig_name"]
                        if ok:
                            grows, leaking = calibrate_removal_boxes(pl["path"], ok, pl["frame_w"], pl["frame_h"],
                                                                     pl["usable"][0], pl["usable"][1],
                                                                     args.watermark_total_area_pct / 100.0)
                            pl["wms"], pl["grows"], pl["wm_leaking"] = ok, grows, leaking
                            pl["wm_source"] = "batch_consensus_verified"
                            pl["notes"].append("own detection found nothing; used verified batch-consensus position")
                            print(f"  {name}: applied verified consensus watermark region(s).")
                        else:
                            pl["notes"].append("no watermark detected and the batch-consensus position "
                                               "is not present in this video")
            else:
                print("\nNo recurring watermark position found across the batch.")

        # STAGE 3: cut the clips from the masters
        n_ok = 0
        for pl in plans:
            if budget_s and time.time() - t_start > budget_s:
                _defer(pl["orig_name"], "deferred: editor time budget reached before this video was rendered (retry next run)")
                n_failed += 1
                if pl.get("master_path") and os.path.exists(pl["master_path"]):
                    os.remove(pl["master_path"])
                continue
            try:
                written = execute_plan(pl)
                if written or args.dry_run:
                    n_ok += 1
                else:
                    n_failed += 1
            except Exception as e:   # noqa: BLE001
                print(f"  !! ERROR rendering {pl['orig_name']} -- continuing with the batch.")
                traceback.print_exc()
                REPORT.append(dict(name=pl["orig_name"], parts_planned=len(pl["parts"]), written=0,
                                   wm_source=pl["wm_source"], wm_regions=len(pl["wms"]),
                                   notes=[f"crashed while rendering: {str(e).strip()[-200:]}"]))
                n_failed += 1

        print(f"\nBatch summary: {n_ok} video(s) produced output, {n_failed} produced none.")
        print_report()
        write_report_json(args.report_json)
        if n_ok == 0 and n_failed > 0 and not args.dry_run:
            sys.exit(1)
    finally:
        shutil.rmtree(master_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
