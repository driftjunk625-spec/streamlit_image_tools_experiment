
# flipbook_app.py
# Upload a GIF → extract frames → impose onto A4 with binding strip, gutters, outer margins,
# cut marks, fixed image sizing (by scale or absolute width), and flush placement.
#
# Key guarantees:
# - Changing the binding margin NEVER resizes the image.
# - Image size is controlled ONLY by user (scale% or width mm).
# - Image is placed flush to the NON-binding edge (optional flush distance control).
# - If things don't fit, the grid (rows/cols) changes—not the image.

import io
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from PIL import Image, ImageSequence, ImageDraw

# ----------------------- Units -----------------------
MM_PER_INCH = 25.4
A4_MM = (210.0, 297.0)  # portrait (w, h)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / MM_PER_INCH))


def px_to_mm(px: int, dpi: int) -> float:
    return px * MM_PER_INCH / dpi


# ----------------------- Specs -----------------------
@dataclass
class PageSpec:
    width_mm: float
    height_mm: float
    dpi: int

    @property
    def size_px(self) -> Tuple[int, int]:
        return (mm_to_px(self.width_mm, self.dpi), mm_to_px(self.height_mm, self.dpi))


@dataclass
class LayoutSpec:
    page_margin_mm: float            # outer page margin (cut marks/crop space)
    gutter_mm: float                 # space between cells (cut channels)
    binding_margin_mm: float         # blank strip on binding side (added to cell width only)
    flip_from_right: bool            # True => binding on left, pages flip from right
    # Image sizing is fixed by user; binding/padding must NOT resize image.
    target_image_width_mm: float | None  # image width in mm (height by aspect) OR
    scale_percent: float | None          # scale% vs native at page DPI
    # Placement refinement
    flush_offset_mm: float           # distance from non-binding edge to image edge (0 = perfectly flush)
    v_align: str                     # 'Top' | 'Center' | 'Bottom'


@dataclass
class Grid:
    cols: int
    rows: int
    img_w_mm: float
    img_h_mm: float

    @property
    def per_page(self) -> int:
        return self.cols * self.rows


# ----------------------- GIF helpers -----------------------
def extract_frames(img: Image.Image, step: int = 1, max_frames: int | None = None) -> List[Image.Image]:
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i % step != 0:
            continue
        f = frame.convert("RGBA")
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    return frames


# ----------------------- Sizing & grid -----------------------
def compute_image_size_mm(frame_w: int, frame_h: int, pagespec: PageSpec, layout: LayoutSpec) -> Tuple[float, float]:
    """Return the IMAGE size in mm based ONLY on user choice (width mm or scale%)."""
    if layout.target_image_width_mm is not None:
        img_w_mm = layout.target_image_width_mm
        aspect = frame_h / frame_w
        img_h_mm = img_w_mm * aspect
    else:
        # scale% relative to image rendered at page DPI
        frame_w_mm_at_dpi = px_to_mm(frame_w, pagespec.dpi)
        img_w_mm = frame_w_mm_at_dpi * (layout.scale_percent / 100.0)
        aspect = frame_h / frame_w
        img_h_mm = img_w_mm * aspect
    return img_w_mm, img_h_mm


def compute_grid(pagespec: PageSpec, layout: LayoutSpec, img_w_mm: float, img_h_mm: float) -> Grid:
    # Available area inside outer page margins
    avail_w = pagespec.width_mm - 2 * layout.page_margin_mm
    avail_h = pagespec.height_mm - 2 * layout.page_margin_mm

    # Each cell = binding strip (on binding side only) + image
    cell_w = img_w_mm + layout.binding_margin_mm
    cell_h = img_h_mm

    def max_fit(total_mm: float, block_mm: float, gap_mm: float) -> int:
        if block_mm <= 0:
            return 0
        return int((total_mm + gap_mm) // (block_mm + gap_mm))

    cols = max_fit(avail_w, cell_w, layout.gutter_mm)
    rows = max_fit(avail_h, cell_h, layout.gutter_mm)
    return Grid(cols, rows, img_w_mm, img_h_mm)


# ----------------------- Cut marks -----------------------
def draw_cut_marks(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid):
    W, H = pagespec.size_px
    margin_px = mm_to_px(layout.page_margin_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.img_w_mm + layout.binding_margin_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.img_h_mm, pagespec.dpi)

    x0, y0 = margin_px, margin_px
    tick_len = mm_to_px(3, pagespec.dpi)
    stroke = max(1, mm_to_px(0.2, pagespec.dpi))
    color = (0, 0, 0, 255)

    # Interior vertical cuts (between columns): ticks in top and bottom margins
    for c in range(1, grid.cols):
        x_cut = x0 + c * cell_w_px + (c - 0) * gutter_px
        draw.line([(x_cut, 0), (x_cut, tick_len)], fill=color, width=stroke)
        draw.line([(x_cut, H - tick_len), (x_cut, H)], fill=color, width=stroke)

    # Interior horizontal cuts (between rows): ticks in left and right margins
    for r in range(1, grid.rows):
        y_cut = y0 + r * cell_h_px + (r - 0) * gutter_px
        draw.line([(0, y_cut), (tick_len, y_cut)], fill=color, width=stroke)
        draw.line([(W - tick_len, y_cut), (W, y_cut)], fill=color, width=stroke)

    # L-shaped corner marks at the four corners of the content rectangle (inside margins)
    cx0, cy0 = x0, y0
    cx1, cy1 = W - margin_px, H - margin_px
    # top-left
    draw.line([(cx0, cy0 - tick_len), (cx0, cy0 + tick_len)], fill=color, width=stroke)
    draw.line([(cx0 - tick_len, cy0), (cx0 + tick_len, cy0)], fill=color, width=stroke)
    # top-right
    draw.line([(cx1, cy0 - tick_len), (cx1, cy0 + tick_len)], fill=color, width=stroke)
    draw.line([(cx1 - tick_len, cy0), (cx1 + tick_len, cy0)], fill=color, width=stroke)
    # bottom-left
    draw.line([(cx0, cy1 - tick_len), (cx0, cy1 + tick_len)], fill=color, width=stroke)
    draw.line([(cx0 - tick_len, cy1), (cx0 + tick_len, cy1)], fill=color, width=stroke)
    # bottom-right
    draw.line([(cx1, cy1 - tick_len), (cx1, cy1 + tick_len)], fill=color, width=stroke)
    draw.line([(cx1 - tick_len, cy1), (cx1 + tick_len, cy1)], fill=color, width=stroke)


# ----------------------- Page composition -----------------------
def compose_pages(
    frames: List[Image.Image],
    pagespec: PageSpec,
    layout: LayoutSpec,
    grid: Grid,
    reverse_order: bool = False,
) -> List[Image.Image]:
    W, H = pagespec.size_px
    margin_px = mm_to_px(layout.page_margin_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    bind_px = mm_to_px(layout.binding_margin_mm, pagespec.dpi)
    flush_px = mm_to_px(layout.flush_offset_mm, pagespec.dpi)

    img_w_px = mm_to_px(grid.img_w_mm, pagespec.dpi)
    img_h_px = mm_to_px(grid.img_h_mm, pagespec.dpi)
    cell_w_px = img_w_px + bind_px
    cell_h_px = img_h_px

    seq = list(reversed(frames)) if reverse_order else list(frames)
    pages: List[Image.Image] = []
    idx = 0
    stroke = max(1, mm_to_px(0.2, pagespec.dpi))

    while idx < len(seq):
        page = Image.new("RGBA", (W, H), (255, 255, 255, 255))
        draw = ImageDraw.Draw(page)

        for r in range(grid.rows):
            for c in range(grid.cols):
                if idx >= len(seq):
                    break
                fx = seq[idx]
                idx += 1

                x_cell = margin_px + c * (cell_w_px + gutter_px)
                y_cell = margin_px + r * (cell_h_px + gutter_px)

                # Resize each frame to the fixed image size
                fr = fx.resize((img_w_px, img_h_px), Image.LANCZOS)

                # Horizontal placement: flush to NON-binding edge with optional offset
                if layout.flip_from_right:
                    # binding on left → non-binding edge is right
                    cx = x_cell + cell_w_px - flush_px - img_w_px
                else:
                    # binding on right → non-binding edge is left
                    cx = x_cell + flush_px

                # Vertical placement
                if layout.v_align == "Top":
                    cy = y_cell
                elif layout.v_align == "Bottom":
                    cy = y_cell + (cell_h_px - img_h_px)
                else:  # Center
                    cy = y_cell + (cell_h_px - img_h_px) // 2

                page.paste(fr, (cx, cy), fr)

                # Optional faint cell border (useful for alignment checks)
                draw.rectangle(
                    [x_cell, y_cell, x_cell + cell_w_px, y_cell + cell_h_px],
                    outline=(0, 0, 0, 30), width=stroke,
                )

        draw_cut_marks(draw, pagespec, layout, grid)
        pages.append(page.convert("RGB"))

    return pages


def save_pages_to_pdf(pages: List[Image.Image]) -> bytes:
    if not pages:
        return b""
    buf = io.BytesIO()
    first, rest = pages[0], pages[1:]
    first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()


# ----------------------- UI -----------------------
st.set_page_config(page_title="Flipbook GIF → A4 PDF", page_icon="📘", layout="centered")
st.title("📘 Flipbook GIF → A4 PDF")
st.caption("Fixed image sizing, true binding strip, precise flush placement, and cut marks for guillotine cutting.")

with st.sidebar:
    st.header("Page & print")
    dpi = st.slider("DPI (print quality)", 150, 600, 300, step=25)
    orient = st.radio("A4 orientation", ["Portrait", "Landscape"], index=0)
    page_w_mm, page_h_mm = A4_MM if orient == "Portrait" else (A4_MM[1], A4_MM[0])

    margin_mm = st.number_input("Outer page margin (mm)", 0.0, 40.0, 8.0, 0.5)
    gutter_mm = st.number_input("Gutter / cut channel between cells (mm)", 0.0, 20.0, 2.0, 0.5)
    binding_mm = st.number_input("Binding strip width (mm)", 0.0, 40.0, 6.0, 0.5)
    flip_from_right = st.radio("Flip direction", ["Flip from right (binding left)", "Flip from left (binding right)"], index=0) == "Flip from right (binding left)"

    st.header("Image sizing (fixed)")
    sizing_mode = st.radio("Sizing mode", ["Scale percent", "Image width (mm)"], index=0)
    target_image_width_mm = None
    scale_percent = None
    if sizing_mode == "Image width (mm)":
        target_image_width_mm = st.number_input("Image width (mm)", min_value=5.0, value=45.0, step=0.5)
    else:
        scale_percent = st.slider("Scale vs. original frame (%)", 10, 400, 100, step=5)

    st.header("Placement")
    flush_offset_mm = st.number_input("Flush distance from NON-binding edge (mm)", min_value=0.0, value=0.0, step=0.5)
    v_align = st.radio("Vertical alignment", ["Top", "Center", "Bottom"], index=1)

    st.header("Frames")
    step = st.number_input("Use every Nth frame", min_value=1, value=1, step=1)
    max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=0, step=1)
    reverse_order = st.checkbox("Reverse order", value=False)

uploaded = st.file_uploader("Upload an animated GIF", type=["gif"], accept_multiple_files=False)

if uploaded is not None:
    try:
        raw = Image.open(uploaded)
    except Exception as e:
        st.error(f"Could not open GIF: {e}")
        st.stop()

    all_frames = extract_frames(raw, step=step, max_frames=(None if max_frames == 0 else max_frames))
    if not all_frames:
        st.warning("No frames extracted with the current settings.")
        st.stop()

    st.success(f"Extracted {len(all_frames)} frame(s)")

    fw, fh = all_frames[0].size
    pagespec = PageSpec(page_w_mm, page_h_mm, dpi)
    layout = LayoutSpec(
        page_margin_mm=margin_mm,
        gutter_mm=gutter_mm,
        binding_margin_mm=binding_mm,
        flip_from_right=flip_from_right,
        target_image_width_mm=target_image_width_mm,
        scale_percent=scale_percent,
        flush_offset_mm=flush_offset_mm,
        v_align=v_align,
    )

    # Compute fixed image size -> derive grid
    img_w_mm, img_h_mm = compute_image_size_mm(fw, fh, pagespec, layout)
    grid = compute_grid(pagespec, layout, img_w_mm, img_h_mm)

    if grid.cols == 0 or grid.rows == 0:
        st.error("Image size + margins/gutter/binding do not fit on A4 at this DPI. Reduce image width/scale or margins, or switch orientation.")
        st.stop()

    # Summary
    st.subheader("Layout summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cards per page", grid.per_page)
        st.write(f"{grid.cols} × {grid.rows}")
    with col2:
        st.write(f"Image size: **{img_w_mm:.1f} × {img_h_mm:.1f} mm**")
        st.write(f"Binding strip: **{binding_mm:.1f} mm** on {'left' if flip_from_right else 'right'}")
    with col3:
        Wpx, Hpx = pagespec.size_px
        st.write(f"A4 @ {dpi} DPI → **{Wpx} × {Hpx} px**")
        st.write(f"Margin: **{margin_mm:.1f}** mm, Gutter: **{gutter_mm:.1f}** mm, Flush: **{flush_offset_mm:.1f}** mm")

    # Compose & preview
    pages = compose_pages(all_frames, pagespec, layout, grid, reverse_order=reverse_order)
    st.subheader("Preview (first page)")
    if pages:
        st.image(pages[0], caption="Page 1 preview", use_column_width=True)

    pdf_bytes = save_pages_to_pdf(pages)
    if pdf_bytes:
        st.download_button(
            label=f"⬇️ Download PDF ({len(pages)} page{'s' if len(pages)!=1 else ''})",
            data=pdf_bytes,
            file_name="flipbook_layout.pdf",
            mime="application/pdf",
        )
else:
    st.info("Upload an animated GIF to begin.")
