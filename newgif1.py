import io
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageSequence
import imageio.v2 as imageio

# --- Streamlit page config ---
st.set_page_config(
    page_title="GIF Frame Splitter & Optimization Check",
    page_icon="🎞️",
    layout="centered",
)

st.title("🎞️ GIF Frame Splitter & Optimization Checker")
st.write(
    "Upload a GIF to extract frames, inspect metadata, and get a heuristic verdict on whether the GIF was likely optimized (delta/partial updates) or not."
)


# --- Data structures ---
@dataclass
class ChangeStats:
    mean_change_ratio: float
    median_change_ratio: float
    small_change_fraction: float
    per_frame_change: List[float]


# --- Helpers ---
@st.cache_data(show_spinner=False)
def load_gif_bytes(data: bytes) -> Tuple[List[np.ndarray], dict]:
    """Decode a GIF to a list of full frames (ndarrays) and return imageio metadata.

    imageio generally yields fully composited frames for GIF, which is what we want for
    change analysis and clean export.
    """
    mem = io.BytesIO(data)
    reader = imageio.get_reader(mem, format="GIF")
    frames: List[np.ndarray] = []
    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}
    for frame in reader:
        frames.append(frame)
    reader.close()
    return frames, meta


@st.cache_data(show_spinner=False)
def pil_info(data: bytes) -> Tuple[Image.Image, dict]:
    """Get first PIL frame and info dict safely."""
    im = Image.open(io.BytesIO(data))
    info = dict(im.info) if getattr(im, "info", None) else {}
    return im.copy(), info


def to_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] == 4:
        return arr
    a = np.full((*arr.shape[:-1], 1), 255, dtype=arr.dtype)
    return np.concatenate([arr, a], axis=-1)


def compute_change_stats(frames: List[np.ndarray]) -> ChangeStats:
    if len(frames) <= 1:
        return ChangeStats(0.0, 0.0, 0.0, [0.0] * len(frames))

    frames_rgba = [to_rgba(f) for f in frames]
    H, W = frames_rgba[0].shape[:2]
    total_pixels = H * W

    per_change: List[float] = []
    for i in range(1, len(frames_rgba)):
        prev = frames_rgba[i - 1].astype(np.int16)
        cur = frames_rgba[i].astype(np.int16)
        diff = np.any(prev != cur, axis=-1)
        changed = int(np.count_nonzero(diff))
        per_change.append(changed / total_pixels)

    small_change_fraction = float(np.mean([r < 0.5 for r in per_change])) if per_change else 0.0
    mean_r = float(np.mean(per_change)) if per_change else 0.0
    median_r = float(np.median(per_change)) if per_change else 0.0
    return ChangeStats(mean_r, median_r, small_change_fraction, per_change)


def estimate_palette_size(first_frame_pil: Image.Image) -> Optional[int]:
    try:
        pal_img = first_frame_pil.convert("P")
        palette = pal_img.getpalette()
        return len(palette) // 3 if palette else None
    except Exception:
        return None


def verdict_from_stats(stats: ChangeStats, palette_size: Optional[int], duration_ms: Optional[int], n_frames: int) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if n_frames <= 1:
        return "Static (single-frame) GIF", ["Only one frame detected."]

    mean_r, med_r, small_frac = stats.mean_change_ratio, stats.median_change_ratio, stats.small_change_fraction

    if mean_r > 0.85 and med_r > 0.85 and small_frac < 0.25:
        verdict = "Likely NOT optimized (full-frame updates)"
        reasons.append(f"Large inter-frame changes (mean {mean_r:.2f}, median {med_r:.2f}).")
    elif mean_r < 0.50 or small_frac > 0.50:
        verdict = "Likely optimized (delta/partial updates)"
        reasons.append(
            f"Smaller inter-frame changes (mean {mean_r:.2f}, median {med_r:.2f}); {small_frac*100:.0f}% frames change <50% area."
        )
    else:
        verdict = "Possibly optimized"
        reasons.append(f"Mixed change profile (mean {mean_r:.2f}, median {med_r:.2f}).")

    if palette_size is not None:
        reasons.append(f"Palette entries: ~{palette_size} (GIF uses up to 256.")
    if duration_ms is not None:
        reasons.append(f"Approx. total duration: {duration_ms} ms.")

    return verdict, reasons


def build_zip_from_frames(frames: List[np.ndarray], prefix: str = "frame") -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, arr in enumerate(frames):
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(f"{prefix}_{i:04d}.png", buf.getvalue())
    mem.seek(0)
    return mem.read()


# --- UI ---
uploaded = st.file_uploader("Upload a .gif file", type=["gif"]) 

if uploaded is not None:
    data = uploaded.read()

    # Preview
    st.subheader("Preview")
    st.image(data, caption="Uploaded GIF", use_column_width=True)

    # Decode + meta
    frames, meta = load_gif_bytes(data)
    first_pil, info = pil_info(data)

    n_frames = len(frames)
    H, W = (frames[0].shape[0], frames[0].shape[1]) if n_frames else (None, None)

    # Metadata
    with st.expander("Metadata"):
        col1, col2 = st.columns(2)
        with col1:
            # Frame duration handling: could be a list or scalar depending on encoder
            durations = meta.get("duration")
            if isinstance(durations, list):
                total_duration_ms = int(sum(durations))
            elif isinstance(durations, (int, float)):
                total_duration_ms = int(durations) * max(n_frames, 1)
            else:
                total_duration_ms = None

            st.write(
                {
                    "width": W,
                    "height": H,
                    "frames": n_frames,
                    "loop": meta.get("loop", info.get("loop")),
                    "duration_per_frame_ms": durations,
                    "total_duration_ms": total_duration_ms,
                }
            )
        with col2:
            palette_size = estimate_palette_size(first_pil)
            st.write(
                {
                    "mode": first_pil.mode,
                    "transparency": info.get("transparency"),
                    "background": info.get("background"),
                    "disposal": info.get("disposal"),
                    "palette_size_estimate": palette_size,
                }
            )

    # Optimization analysis
    st.subheader("Optimization check")
    if n_frames <= 1:
        st.info("This GIF has only one frame, so optimization analysis doesn't apply.")
    else:
        stats = compute_change_stats(frames)
        # Compute total duration again for verdict reasons
        durations = meta.get("duration")
        if isinstance(durations, list):
            total_duration_ms = int(sum(durations))
        elif isinstance(durations, (int, float)):
            total_duration_ms = int(durations) * n_frames
        else:
            total_duration_ms = None

        verdict, reasons = verdict_from_stats(stats, palette_size, total_duration_ms, n_frames)

        st.markdown(f"**Verdict:** {verdict}")
        st.markdown("**Why:**\n- " + "\n- ".join(reasons))

        with st.expander("Inter-frame change ratios (0=no change, 1=full frame changed)"):
            st.json(
                {
                    "mean_change_ratio": round(stats.mean_change_ratio, 4),
                    "median_change_ratio": round(stats.median_change_ratio, 4),
                    "small_change_fraction": round(stats.small_change_fraction, 4),
                    "per_frame_change": [round(x, 4) for x in stats.per_frame_change],
                }
            )

    # Frames grid
    st.subheader("Extracted frames")
    grid_cols = st.slider("Columns in preview grid", 2, 8, 5)
    if frames:
        rows = (len(frames) + grid_cols - 1) // grid_cols
        for r in range(rows):
            cols = st.columns(grid_cols)
            for c in range(grid_cols):
                idx = r * grid_cols + c
                if idx < len(frames):
                    with cols[c]:
                        st.image(frames[idx], caption=f"#{idx}", use_column_width=True)

        # Downloads
        st.subheader("Download frames")
        with st.spinner("Preparing ZIP of PNG frames..."):
            zip_bytes = build_zip_from_frames(frames)
        st.download_button(
            "Download frames.zip",
            data=zip_bytes,
            file_name="frames.zip",
            mime="application/zip",
        )

# Footer
st.markdown("---")
st.caption(
    "'Likely optimized' means many frames only change small regions vs. the canvas; 'Likely NOT optimized' means most frames are full-frame updates."
)
