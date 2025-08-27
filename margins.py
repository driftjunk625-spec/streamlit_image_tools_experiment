# Print-a-GIF — Streamlit App (Python)
# Inspired by stupotmcdoodlepip/Print-A-Gif (C#) — reimplemented as a
# fully customizable Streamlit app to split a GIF into frames and lay
# them out as printable A4 pages for an easy-to-cut flipbook.
#
# Working draft based on Version B, updated per request:
# - Grid layout only (multiple frames per sheet)
# - A4 + Custom page sizes
# - Lock original scale (100%)
# - Frame scale (%) slider (10–200)
# - Left-side only outer spacing (flush right edge when binding left)
# - Cut marks for **every frame** (not just outer grid)
# - Per-frame cutting margin on all four sides + extra binding offset on chosen side
#   → You cut three normal edges and one binding edge per frame

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageSequence
from reportlab.pdfgen import canvas
import streamlit as st

# ------------------------------- Utilities ------------------------------- #

MM_PER_INCH = 25.4


def mm_to_px(mm_val: float, dpi: float) -> int:
    return int(round(mm_val / MM_PER_INCH * dpi))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    raise ValueError("Expected 6-digit hex color like #RRGGBB")


@dataclass
class PageSpec:
    width_mm: float
    height_mm: float
    orientation: str  # 'portrait' | 'landscape'
    dpi: int

    @property
    def width_px(self) -> int:
        w_mm, h_mm = (self.width_mm, self.height_mm)
        if self.orientation == 'landscape':
            w_mm, h_mm = (self.height_mm, self.width_mm)
        return mm_to_px(w_mm, self.dpi)

    @property
    def height_px(self) -> int:
        w_mm, h_mm = (self.width_mm, self.height_mm)
        if self.orientation == 'landscape':
            w_mm, h_mm = (self.height_mm, self.width_mm)
        return mm_to_px(h_mm, self.dpi)

    @property
    def pagesize_pts(self) -> Tuple[float, float]:
        # For ReportLab PDF canvas (points = 1/72")
        w_in = (self.width_mm / MM_PER_INCH)
        h_in = (self.height_mm / MM_PER_INCH)
        if self.orientation == 'landscape':
            w_in, h_in = h_in, w_in
        return (w_in * 72.0, h_in * 72.0)


# ------------------------- GIF Frame Extraction -------------------------- #


def coalesce_gif(im: Image.Image) -> List[Image.Image]:
    """Return list of RGBA frames composited ("unoptimised")."""
    frames: List[Image.Image] = []
    previous = Image.new('RGBA', im.size, (0, 0, 0, 0))
    palette = im.getpalette()

    for frame in ImageSequence.Iterator(im):
        if frame.mode == 'P':
            if not frame.getpalette() and palette is not None:
                frame.putpalette(palette)
            else:
                palette = frame.getpalette()
            frame_rgba = frame.convert('RGBA')
        else:
            frame_rgba = frame.convert('RGBA')
        composed = Image.alpha_composite(previous, frame_rgba)
        frames.append(composed)
        previous = composed

    return frames


def extract_gif_frames(file_bytes: bytes, unoptimize: bool) -> List[Image.Image]:
    with Image.open(io.BytesIO(file_bytes)) as im:
        if unoptimize:
            return coalesce_gif(im)
        else:
            return [f.convert('RGBA') for f in ImageSequence.Iterator(im)]


# --------------------------- Layout Composition -------------------------- #

@dataclass
class LayoutParams:
    cols: int
    rows: int
    print_margin_mm: float          # outer page margin (outside grid)
    cut_spacing_mm: float           # spacing between cells (gutters)
    tab_width_mm: float             # binding tab on selected page edge
    tab_side: str                   # 'left' | 'right' | 'top' | 'bottom' | 'none'
    spacing_hex: str                # background/gutter color
    draw_guides: bool
    draw_numbers: bool
    number_font_size_pt: int

    # Frame sizing
    lock_original_scale: bool = field(default=False)
    frame_scale_pct: int = field(default=100)        # 10–200

    # Outer spacing option
    left_only_outer_spacing: bool = field(default=False)

    # Per-frame cutting box
    frame_margin_mm: float = field(default=3.0)      # equal on all 4 sides
    frame_binding_extra_mm: float = field(default=3.0)  # extra only on binding side per frame

    # Cut marks (per frame + outer content box)
    show_cut_marks: bool = field(default=True)
    cut_mark_len_mm: float = field(default=4.0)
    cut_mark_offset_mm: float = field(default=0.0)
    cut_mark_width_px: int = field(default=1)


@dataclass
class FrameSelection:
    start: int  # inclusive, 0-based
    end: int    # exclusive, -1 -> to the end
    step: int   # 1 = every frame, 2 = every other, etc.


@dataclass
class BuildArtifacts:
    frames_rgba: List[Image.Image]
    page_images_rgb: List[Image.Image]
    pdf_bytes: bytes
    frames_zip: bytes
    pages_zip: bytes


def compose_pages(frames: List[Image.Image], page: PageSpec, layout: LayoutParams) -> List[Image.Image]:
    W, H = page.width_px, page.height_px
    margin_px = mm_to_px(layout.print_margin_mm, page.dpi)
    spacing_px = mm_to_px(layout.cut_spacing_mm, page.dpi)
    tab_px = mm_to_px(layout.tab_width_mm, page.dpi) if layout.tab_side != 'none' else 0

    # Asymmetric outer spacing support (primarily for left binding)
    margin_left = margin_px
    margin_right = margin_px  # symmetric; binding handled per-frame
    margin_top = margin_px
    margin_bottom = margin_px

    # Content area after margins
    content_x = margin_left
    content_y = margin_top
    content_w = W - (margin_left + margin_right)
    content_h = H - (margin_top + margin_bottom)

    # Reserve binding tab on the chosen side (outside the grid area)
    if layout.tab_side == 'left':
        content_x += tab_px
        content_w -= tab_px
    elif layout.tab_side == 'right':
        content_w -= tab_px
    elif layout.tab_side == 'top':
        content_y += tab_px
        content_h -= tab_px
    elif layout.tab_side == 'bottom':
        content_h -= tab_px

    cols = max(1, layout.cols)
    rows = max(1, layout.rows)

    # Compute cell size available for each frame, accounting for spacings
    cell_w = (content_w - (cols - 1) * spacing_px) / cols
    cell_h = (content_h - (rows - 1) * spacing_px) / rows

    # Integer-perfect grid bounds so last col/row snaps to right/bottom
    spacing_int = int(spacing_px)

    col_bounds: List[Tuple[int, int]] = []
    x = int(content_x)
    for c in range(cols):
        if c == cols - 1:
            x1 = int(content_x + content_w)
        else:
            x1 = x + int(round(cell_w))
        col_bounds.append((x, x1))
        x = x1 + spacing_int

    row_bounds: List[Tuple[int, int]] = []
    y = int(content_y)
    for r in range(rows):
        if r == rows - 1:
            y1 = int(content_y + content_h)
        else:
            y1 = y + int(round(cell_h))
        row_bounds.append((y, y1))
        y = y1 + spacing_int

    # Base background page color = spacing color
    bg_rgb = hex_to_rgb(layout.spacing_hex)

    page_images: List[Image.Image] = []

    def scale_frame_to_rect(im: Image.Image, rw: int, rh: int) -> Image.Image:
        if layout.lock_original_scale:
            return im  # no scaling
        iw, ih = im.size
        fit_scale = min(rw / iw, rh / ih)
        user_scale = max(0.1, min(2.0, layout.frame_scale_pct / 100.0))
        scale = fit_scale * user_scale
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        return im.resize((new_w, new_h), Image.LANCZOS)

    def draw_outer_tab(page_draw: ImageDraw.ImageDraw):
        if layout.tab_side == 'left':
            page_draw.rectangle([margin_left - tab_px, margin_top, margin_left, H - margin_bottom], fill=bg_rgb)
        elif layout.tab_side == 'right':
            page_draw.rectangle([W - margin_right, margin_top, W - margin_right + tab_px, H - margin_bottom], fill=bg_rgb)
        elif layout.tab_side == 'top':
            page_draw.rectangle([margin_left, margin_top - tab_px, W - margin_right, margin_top], fill=bg_rgb)
        elif layout.tab_side == 'bottom':
            page_draw.rectangle([margin_left, H - margin_bottom, W - margin_right, H - margin_bottom + tab_px], fill=bg_rgb)

    def draw_cut_ticks_for_rect(draw: ImageDraw.ImageDraw, rx0: int, ry0: int, rx1: int, ry1: int):
        if not layout.show_cut_marks:
            return
        L = mm_to_px(layout.cut_mark_len_mm, page.dpi)
        off = mm_to_px(layout.cut_mark_offset_mm, page.dpi)
        w = layout.cut_mark_width_px
        # Four edges: draw short ticks at corners
        # Top edge
        draw.line([(rx0, ry0 - off), (rx0 + L, ry0 - off)], fill=(0, 0, 0), width=w)
        draw.line([(rx1 - L, ry0 - off), (rx1, ry0 - off)], fill=(0, 0, 0), width=w)
        # Bottom edge
        draw.line([(rx0, ry1 + off), (rx0 + L, ry1 + off)], fill=(0, 0, 0), width=w)
        draw.line([(rx1 - L, ry1 + off), (rx1, ry1 + off)], fill=(0, 0, 0), width=w)
        # Left edge
        draw.line([(rx0 - off, ry0), (rx0 - off, ry0 + L)], fill=(0, 0, 0), width=w)
        draw.line([(rx0 - off, ry1 - L), (rx0 - off, ry1)], fill=(0, 0, 0), width=w)
        # Right edge
        draw.line([(rx1 + off, ry0), (rx1 + off, ry0 + L)], fill=(0, 0, 0), width=w)
        draw.line([(rx1 + off, ry1 - L), (rx1 + off, ry1)], fill=(0, 0, 0), width=w)

    # Pagination
    frames_iter = iter(frames)

    while True:
        page_img = Image.new('RGB', (W, H), color=bg_rgb)
        page_draw = ImageDraw.Draw(page_img)

        # Draw outer binding tab (page-level)
        draw_outer_tab(page_draw)

        placed_any = False

        for r in range(rows):
            for c in range(cols):
                try:
                    frame = next(frames_iter)
                except StopIteration:
                    if not placed_any:
                        return page_images
                    else:
                        page_images.append(page_img)
                        return page_images

                placed_any = True

                # Cell bounds
                y0, y1 = row_bounds[r]
                x0, x1 = col_bounds[c]

                # Per-frame margins and binding extra (inside the cell)
                fm = mm_to_px(layout.frame_margin_mm, page.dpi)
                fbind = mm_to_px(layout.frame_binding_extra_mm, page.dpi)

                # Compute the 4 cut lines around the image area
                left_cut = x0 + fm + (fbind if layout.tab_side == 'left' else 0)
                right_cut = x1 - fm - (fbind if layout.tab_side == 'right' else 0)
                top_cut = y0 + fm + (fbind if layout.tab_side == 'top' else 0)
                bottom_cut = y1 - fm - (fbind if layout.tab_side == 'bottom' else 0)

                # Ensure minimum viable area
                if right_cut <= left_cut or bottom_cut <= top_cut:
                    continue

                # Draw per-frame cut marks on the 4 cut lines
                draw_cut_ticks_for_rect(page_draw, left_cut, top_cut, right_cut, bottom_cut)

                # Place the frame inside the inner rect (left_cut..right_cut, top_cut..bottom_cut)
                rw, rh = right_cut - left_cut, bottom_cut - top_cut
                fr = scale_frame_to_rect(frame, rw, rh)
                fw, fh = fr.size

                # Align against the opposite edge of the binding side:
                # - Binding LEFT  → image flush RIGHT
                # - Binding RIGHT → image flush LEFT
                # - Binding TOP   → image flush BOTTOM
                # - Binding BOTTOM→ image flush TOP
                if layout.tab_side == 'left':
                    paste_x = right_cut - fw
                    paste_y = top_cut + (rh - fh) // 2
                elif layout.tab_side == 'right':
                    paste_x = left_cut
                    paste_y = top_cut + (rh - fh) // 2
                elif layout.tab_side == 'top':
                    paste_x = left_cut + (rw - fw) // 2
                    paste_y = bottom_cut - fh
                elif layout.tab_side == 'bottom':
                    paste_x = left_cut + (rw - fw) // 2
                    paste_y = top_cut
                else:
                    paste_x = left_cut + (rw - fw) // 2
                    paste_y = top_cut + (rh - fh) // 2

                # Crop to the inner rect if overflow
                if layout.lock_original_scale or fw > rw or fh > rh:
                    fx0, fy0, fx1, fy1 = paste_x, paste_y, paste_x + fw, paste_y + fh
                    ox0, oy0 = max(fx0, left_cut), max(fy0, top_cut)
                    ox1, oy1 = min(fx1, right_cut), min(fy1, bottom_cut)
                    if ox1 > ox0 and oy1 > oy0:
                        src_x0, src_y0 = ox0 - fx0, oy0 - fy0
                        src_x1, src_y1 = src_x0 + (ox1 - ox0), src_y0 + (oy1 - oy0)
                        fr_cropped = fr.crop((src_x0, src_y0, src_x1, src_y1)).convert('RGB')
                        page_img.paste(fr_cropped, (ox0, oy0))
                else:
                    page_img.paste(fr.convert('RGB'), (paste_x, paste_y))

                # Optional frame number in the bottom-right of the inner rect
                if layout.draw_numbers:
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                    label = str(len(page_images) * (rows * cols) + r * cols + c + 1)
                    tx = right_cut - 6
                    ty = bottom_cut - 12
                    page_draw.text((tx + 1, ty + 1), label, fill=(255, 255, 255), anchor='rs', font=font)
                    page_draw.text((tx, ty), label, fill=(0, 0, 0), anchor='rs', font=font)

        # Finished a full page
        page_images.append(page_img)


# ------------------------------- PDF Build ------------------------------- #

def build_pdf(pages: List[Image.Image], page: PageSpec) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page.pagesize_pts)

    for p in pages:
        img_buf = io.BytesIO()
        p.save(img_buf, format='PNG')
        img_buf.seek(0)
        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(img_buf), 0, 0, width=page.pagesize_pts[0], height=page.pagesize_pts[1])
        c.showPage()

    c.save()
    buf.seek(0)
    return buf.read()


# ----------------------------- ZIP Artifacts ----------------------------- #

def zip_frames(frames: List[Image.Image]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i, fr in enumerate(frames):
            img_bytes = io.BytesIO()
            fr.save(img_bytes, format='PNG')
            zf.writestr(f'frame_{i:04d}.png', img_bytes.getvalue())
    mem.seek(0)
    return mem.read()


def zip_pages(pages: List[Image.Image]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for i, pg in enumerate(pages):
            img_bytes = io.BytesIO()
            pg.save(img_bytes, format='PNG')
            zf.writestr(f'page_{i:03d}.png', img_bytes.getvalue())
    mem.seek(0)
    return mem.read()


# ------------------------------- Streamlit ------------------------------- #

def main():
    st.set_page_config(page_title="Print-a-GIF (Streamlit)", layout="wide")
    st.title("🖨️ Print‑a‑GIF — Flipbook Maker")
    st.caption("Upload a GIF, tune the layout, and export an A4‑ready PDF.")

    with st.sidebar:
        st.header("1) GIF Input")
        file = st.file_uploader("Choose a GIF", type=["gif"])
        unopt = st.checkbox("Unoptimise first (coalesce frames)", value=False,
                            help="Fixes many 'optimised' GIFs where later frames only store differences.")

        st.header("2) Frame Selection")
        col_fs1, col_fs2, col_fs3 = st.columns(3)
        with col_fs1:
            start = st.number_input("Start frame (0‑based)", min_value=0, value=0, step=1)
        with col_fs2:
            end = st.number_input("End frame (exclusive, -1 for all)", min_value=-1, value=-1, step=1)
        with col_fs3:
            step = st.number_input("Step (every Nth)", min_value=1, value=1, step=1)

        st.header("3) Page & Output")
        dpi = st.slider("DPI", min_value=150, max_value=600, value=300, step=25)
        page_size_opt = st.selectbox("Page size", ["A4 (210×297 mm)", "Custom (mm)"])
        orientation = st.selectbox("Orientation", ["portrait", "landscape"], index=0)
        if page_size_opt.startswith("A4"):
            page_spec = PageSpec(width_mm=210.0, height_mm=297.0, orientation=orientation, dpi=dpi)
        else:
            wmm = st.number_input("Width (mm)", min_value=50.0, max_value=2000.0, value=210.0, step=1.0)
            hmm = st.number_input("Height (mm)", min_value=50.0, max_value=2000.0, value=297.0, step=1.0)
            page_spec = PageSpec(width_mm=wmm, height_mm=hmm, orientation=orientation, dpi=dpi)

        st.header("4) Layout")
        cols = st.number_input("Repeat X (columns)", min_value=1, max_value=12, value=2, step=1)
        rows = st.number_input("Repeat Y (rows)", min_value=1, max_value=40, value=8, step=1)
        margin = st.number_input("Outer page margin (mm)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        spacing = st.number_input("Cell spacing / gutter (mm)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        tab_width = st.number_input("Binding tab width (mm)", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
        tab_side = st.selectbox("Binding tab side (page)", ["none", "left", "right", "top", "bottom"], index=1)        # (Removed) Left-side only outer spacing checkbox; use per-frame binding extra instead.
        spacing_color = st.color_picker("Spacing/background color", value="#FFFF00")

        draw_guides = st.checkbox("Draw cell outlines (debug)", value=False)
        draw_numbers = st.checkbox("Frame numbers", value=False)

        lock_original = st.checkbox(
            "Lock original scale (100%)",
            value=False,
            help="Paste each frame at its native pixel size; page DPI controls physical size. If a frame is larger than its inner box, it will be cropped to that box."
        )
        scale_pct = st.slider(
            "Frame scale (%)", min_value=10, max_value=200, value=100, step=5,
            help="Scale frames inside each frame's inner box. Ignored when 'Lock original scale' is enabled."
        )

        st.subheader("Per-frame cutting box")
        frame_margin = st.number_input("Cutting margin (mm) — all sides", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
        frame_bind_extra = st.number_input("Binding extra on bound side (mm)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)

        st.subheader("Cut marks")
        show_cuts = st.checkbox("Show cut marks", value=True)
        cut_len = st.number_input("Mark length (mm)", min_value=1.0, max_value=20.0, value=4.0, step=0.5)
        cut_off = st.number_input("Mark offset from edge (mm)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)

    if file is None:
        st.info("Upload a GIF in the sidebar to begin.")
        return

    frames = extract_gif_frames(file.read(), unoptimize=unopt)
    total = len(frames)

    if end == -1 or end > total:
        end_eff = total
    else:
        end_eff = end
    sel_idx = list(range(start, end_eff, step))
    frames_sel = [frames[i] for i in sel_idx if 0 <= i < total]

    if tab_width == 10.0 and frames_sel:
        default_tab_mm = max(5.0, (frames_sel[0].size[0] * 0.2) / page_spec.dpi * MM_PER_INCH)
        st.sidebar.caption(f"Suggested tab width ~ {default_tab_mm:.1f} mm (20% of frame width)")

    layout = LayoutParams(
        cols=int(cols),
        rows=int(rows),
        print_margin_mm=float(margin),
        cut_spacing_mm=float(spacing),
        tab_width_mm=float(tab_width),
        tab_side=tab_side,
        spacing_hex=spacing_color,
        draw_guides=bool(draw_guides),
        draw_numbers=bool(draw_numbers),
        number_font_size_pt=8,
        lock_original_scale=bool(lock_original),
        frame_scale_pct=int(scale_pct),
        # left_only_outer_spacing ignored; binding handled per frame
        frame_margin_mm=float(frame_margin),
        frame_binding_extra_mm=float(frame_bind_extra),
        show_cut_marks=bool(show_cuts),
        cut_mark_len_mm=float(cut_len),
        cut_mark_offset_mm=float(cut_off),
    )

    pages = compose_pages(frames_sel, page_spec, layout)

    st.subheader("Preview")
    if pages:
        preview = pages[0].copy()
        preview_w = min(900, preview.size[0])
        scale = preview_w / preview.size[0]
        preview = preview.resize((int(preview.size[0]*scale), int(preview.size[1]*scale)), Image.BILINEAR)
        st.image(preview, caption=f"Page 1 of {len(pages)} (preview)")
    else:
        st.warning("No pages produced. Check your frame selection and layout settings.")
        return

    pdf_bytes = build_pdf(pages, page_spec)
    frames_zip = zip_frames(frames_sel)
    pages_zip = zip_pages(pages)

    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="print_a_gif.pdf", mime="application/pdf")
    with c2:
        st.download_button("⬇️ Frames (ZIP)", data=frames_zip, file_name="frames.zip", mime="application/zip")
    with c3:
        st.download_button("⬇️ Pages as PNG (ZIP)", data=pages_zip, file_name="pages.zip", mime="application/zip")

    st.caption(
        f"Frames used: {len(frames_sel)} • Pages: {len(pages)} • Grid: {layout.cols}×{layout.rows} • DPI: {page_spec.dpi} • Page: {page_spec.width_mm:.0f}×{page_spec.height_mm:.0f} mm ({orientation})"
    )


if __name__ == "__main__":
    main()
