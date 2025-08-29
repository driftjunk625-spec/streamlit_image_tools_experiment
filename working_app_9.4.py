"""
This file was auto-generated as a *commented* companion to the original script.
It preserves the original code but adds an overview and a function/class map to help you understand how it works.

## Quick facts
- Original path: /mnt/data/flipbook_app_v9_4-2.py
- Lines of code: 738
- Characters: 36216
- Detected imports (6):
  - import io
  - import os
  - from dataclasses import dataclass
  - from typing import List, Tuple, Optional
  - import streamlit as st
  - from PIL import Image, ImageSequence, ImageDraw, ImageFont

## Major libraries & roles (guessed)
- Streamlit-based UI detected
- Pillow (PIL) for image handling
- NumPy arrays

## Global constants (heuristic)
- L11: MM_PER_INCH = 25.4
- L13: PAPER_SIZES_MM = {

## Function & class map
- L21: def mm_to_px(mm, dpi) -> int [docstring: no]
- L24: def px_to_mm(px, dpi) -> float [docstring: no]
- L27: def hex_to_rgba(hex_color, alpha) [docstring: no]
- L38: def try_load_font(size_px, user_font_path) [docstring: no]
- L65: class PageSpec [docstring: no]
- L75: def size_px(self) -> Tuple[int, int] [decorators: property] [docstring: no]
- L79: class LayoutSpec [docstring: no]
- L129: class Grid [docstring: no]
- L137: def per_page(self) -> int [decorators: property] [docstring: no]
- L140: def extract_frames(img, step, max_frames) -> List[Image.Image] [docstring: no]
- L150: def compute_image_size_mm(frame_w, frame_h, pagespec, layout) -> Tuple[float, float] [docstring: no]
- L172: def compute_card_size_mm(img_w_mm, img_h_mm, layout) -> Tuple[float, float] [docstring: no]
- L181: def compute_grid(pagespec, layout, card_w_mm, card_h_mm, img_w_mm, img_h_mm) -> 'Grid' [docstring: no]
- L184: def max_fit(total_mm, block_mm, gap_mm) [docstring: no]
- L192: def draw_margin_box(draw, pagespec, color, stroke_px) [docstring: no]
- L200: def draw_card_grid(draw, pagespec, layout, grid, color, stroke_px) [docstring: no]
- L212: def draw_gutter_guides(draw, pagespec, layout, grid, color, stroke_px) [docstring: no]
- L228: def draw_card_cut_marks(draw, pagespec, layout, grid, color, stroke_px) [docstring: no]
- L245: def draw_cell_left_edge_marks_in_margins(draw, pagespec, layout, grid, color, stroke_px) [docstring: no]
- L256: def draw_photo_marks_content(draw, pagespec, layout, img_rects, color, stroke_px) [docstring: no]
- L266: def draw_photo_marks_in_margins(draw, pagespec, layout, img_rects, color, stroke_px) [docstring: no]
- L281: def draw_text_centered(draw, text, center_xy, font, color) [docstring: no]
- L294: def draw_binding_overlays(draw, pagespec, layout, grid, color, stroke_px) [docstring: no]
- L326: def compose_pages(frames, pagespec, layout, grid, reverse_order, base_filename) [docstring: no]
- L429: def save_pages_to_pdf(pages) -> bytes [docstring: no]

## How to read this file
1. Start with this block to understand the structure and purpose.
2. Then scroll into the original code below; it's unchanged from your source.
3. Use the line numbers in the map above to jump to functions/classes of interest.

Auto-generated on: 2025-08-25T23:48:04
"""


# flipbook_app_v9_4.py
import io
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageSequence, ImageDraw, ImageFont

MM_PER_INCH = 25.4

PAPER_SIZES_MM = {
    "A4": (210.0, 297.0),
    "Letter": (215.9, 279.4),
    "A3": (297.0, 420.0),
    "6×4 in": (152.4, 101.6),
    "Custom": None
}

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / MM_PER_INCH))

def px_to_mm(px: int, dpi: int) -> float:
    return px * MM_PER_INCH / dpi

def hex_to_rgba(hex_color: str, alpha: int = 255):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)
    elif len(hex_color) == 3:
        r = int(hex_color[0]*2, 16); g = int(hex_color[1]*2, 16); b = int(hex_color[2]*2, 16)
        return (r, g, b, alpha)
    else:
        return (0, 0, 0, alpha)

def try_load_font(size_px: int, user_font_path: Optional[str] = None):
    if user_font_path and os.path.exists(user_font_path):
        try:
            return (ImageFont.truetype(user_font_path, size_px), True, user_font_path)
        except Exception:
            pass
    candidate_paths = [
        "Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/Library/Fonts/Roboto-Regular.ttf",
        "C:\\Windows\\Fonts\\Roboto-Regular.ttf",
        "Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/local/share/fonts/DejaVuSans.ttf",
    ]
    for p in candidate_paths:
        try:
            return (ImageFont.truetype(p, size_px), True, p)
        except Exception:
            continue
    return (ImageFont.load_default(), False, None)

@dataclass
class PageSpec:
    width_mm: float
    height_mm: float
    dpi: int
    margin_top_mm: float
    margin_right_mm: float
    margin_bottom_mm: float
    margin_left_mm: float

    @property
    def size_px(self) -> Tuple[int, int]:
        return (mm_to_px(self.width_mm, self.dpi), mm_to_px(self.height_mm, self.dpi))

@dataclass
class LayoutSpec:
    gutter_mm: float
    binding_margin_mm: float
    flip_from: str
    binding_side: str
    size_mode: str                 # 'scale','locked','independent'
    target_w_mm: Optional[float]
    target_h_mm: Optional[float]
    scale_percent: Optional[float]
    pad_left_mm: float
    pad_right_mm: float
    pad_top_mm: float
    pad_bottom_mm: float
    # Overlays
    show_margin_box: bool
    show_card_grid: bool
    show_gutter_guides: bool
    show_card_cut_marks: bool
    show_cell_left_edge_marks_margins: bool
    show_binding_strip_overlay: bool
    show_photo_edge_marks_content: bool
    show_photo_edge_marks_margins: bool
    # Geometry/colors
    tick_len_mm: float
    photo_tick_len_mm: float
    tick_thickness_mm: float
    color_card_marks: Tuple[int,int,int,int]
    color_cell_left_marks: Tuple[int,int,int,int]
    color_gutter_guides: Tuple[int,int,int,int]
    color_photo_marks_content: Tuple[int,int,int,int]
    color_photo_marks_margins: Tuple[int,int,int,int]
    color_margin_box: Tuple[int,int,int,int]
    color_binding_overlay: Tuple[int,int,int,int]
    # Labels
    show_frame_numbers_in_binding: bool
    frame_start_number: int
    frame_number_font_mm: float
    frame_number_near_spine: bool
    frame_number_rotate_vertical: bool
    show_filename_in_bottom_margin: bool
    filename_font_mm: float
    filename_color: Tuple[int,int,int,int]
    filename_x_offset_mm: float
    filename_y_offset_mm: float
    # Per-side photo-edge offsets (mm)
    photo_margin_mark_offset_b_mm: float
    photo_margin_mark_offset_d_mm: float
    photo_margin_mark_offset_u_mm: float
    photo_margin_mark_offset_n_mm: float
    # Debug/fonts
    show_image_debug_overlay: bool
    debug_font_mm: float
    user_font_path: Optional[str]

@dataclass
class Grid:
    cols: int
    rows: int
    card_w_mm: float
    card_h_mm: float
    img_w_mm: float
    img_h_mm: float
    @property
    def per_page(self) -> int:
        return self.cols * self.rows

def extract_frames(img: Image.Image, step: int = 1, max_frames: int | None = None) -> List[Image.Image]:
    frames = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i % step != 0:
            continue
        frames.append(frame.convert("RGBA"))
        if max_frames and len(frames) >= max_frames:
            break
    return frames

def compute_image_size_mm(frame_w: int, frame_h: int, pagespec: PageSpec, layout: LayoutSpec) -> Tuple[float, float]:
    aspect = (frame_h / frame_w) if frame_w else 1.0
    if layout.size_mode == "scale":
        frame_w_mm_at_dpi = px_to_mm(frame_w, pagespec.dpi)
        img_w_mm = frame_w_mm_at_dpi * (layout.scale_percent / 100.0)
        img_h_mm = img_w_mm * aspect
    elif layout.size_mode == "locked":
        if layout.target_w_mm is not None:
            img_w_mm = layout.target_w_mm
            img_h_mm = img_w_mm * aspect
        elif layout.target_h_mm is not None:
            img_h_mm = layout.target_h_mm
            img_w_mm = img_h_mm / aspect if aspect else layout.target_h_mm
        else:
            frame_w_mm_at_dpi = px_to_mm(frame_w, pagespec.dpi)
            img_w_mm = frame_w_mm_at_dpi
            img_h_mm = img_w_mm * aspect
    else:
        img_w_mm = layout.target_w_mm if layout.target_w_mm is not None else px_to_mm(frame_w, pagespec.dpi)
        img_h_mm = layout.target_h_mm if layout.target_h_mm is not None else px_to_mm(frame_h, pagespec.dpi)
    return img_w_mm, img_h_mm

def compute_card_size_mm(img_w_mm: float, img_h_mm: float, layout: LayoutSpec) -> Tuple[float, float]:
    w = img_w_mm + layout.pad_left_mm + layout.pad_right_mm
    h = img_h_mm + layout.pad_top_mm + layout.pad_bottom_mm
    if layout.binding_side in ("left", "right"):
        w += layout.binding_margin_mm
    else:
        h += layout.binding_margin_mm
    return w, h

def compute_grid(pagespec: PageSpec, layout: LayoutSpec, card_w_mm, card_h_mm, img_w_mm, img_h_mm) -> 'Grid':
    avail_w = pagespec.width_mm - (pagespec.margin_left_mm + pagespec.margin_right_mm)
    avail_h = pagespec.height_mm - (pagespec.margin_top_mm + pagespec.margin_bottom_mm)
    def max_fit(total_mm, block_mm, gap_mm):
        if block_mm <= 0:
            return 0
        return int((total_mm + gap_mm) // (block_mm + gap_mm))
    cols = max_fit(avail_w, card_w_mm, layout.gutter_mm)
    rows = max_fit(avail_h, card_h_mm, layout.gutter_mm)
    return Grid(cols, rows, card_w_mm, card_h_mm, img_w_mm, img_h_mm)

def draw_margin_box(draw: ImageDraw.ImageDraw, pagespec: PageSpec, color, stroke_px):
    W, H = pagespec.size_px
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    x1 = W - mm_to_px(pagespec.margin_right_mm, pagespec.dpi)
    y1 = H - mm_to_px(pagespec.margin_bottom_mm, pagespec.dpi)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=stroke_px)

def draw_card_grid(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, color, stroke_px):
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.card_h_mm, pagespec.dpi)
    for r in range(grid.rows):
        for c in range(grid.cols):
            x = x0 + c*(cell_w_px + gutter_px)
            y = y0 + r*(cell_h_px + gutter_px)
            draw.rectangle([x, y, x+cell_w_px, y+cell_h_px], outline=color, width=stroke_px)

def draw_gutter_guides(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, color, stroke_px):
    W, H = pagespec.size_px
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    x1 = W - mm_to_px(pagespec.margin_right_mm, pagespec.dpi)
    y1 = H - mm_to_px(pagespec.margin_bottom_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.card_h_mm, pagespec.dpi)
    for c in range(1, grid.cols):
        x_cut = x0 + c*cell_w_px + (c-0)*gutter_px - gutter_px//2
        draw.line([(x_cut, y0), (x_cut, y1)], fill=color, width=stroke_px)
    for r in range(1, grid.rows):
        y_cut = y0 + r*cell_h_px + (r-0)*gutter_px - gutter_px//2
        draw.line([(x0, y_cut), (x1, y_cut)], fill=color, width=stroke_px)

def draw_card_cut_marks(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, color, stroke_px):
    W, H = pagespec.size_px
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.card_h_mm, pagespec.dpi)
    tick_len = mm_to_px(layout.tick_len_mm, pagespec.dpi)
    for c in range(1, grid.cols):
        x_cut = x0 + c*cell_w_px + (c-0)*gutter_px
        draw.line([(x_cut, 0), (x_cut, tick_len)], fill=color, width=stroke_px)
        draw.line([(x_cut, H - tick_len), (x_cut, H)], fill=color, width=stroke_px)
    for r in range(1, grid.rows):
        y_cut = y0 + r*cell_h_px + (r-0)*gutter_px
        draw.line([(0, y_cut), (tick_len, y_cut)], fill=color, width=stroke_px)
        draw.line([(W - tick_len, y_cut), (W, y_cut)], fill=color, width=stroke_px)

def draw_cell_left_edge_marks_in_margins(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, color, stroke_px):
    W, H = pagespec.size_px
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    tick_len = mm_to_px(layout.tick_len_mm, pagespec.dpi)
    for c in range(0, grid.cols):
        x_left = x0 + c*(cell_w_px + gutter_px)
        draw.line([(x_left, 0), (x_left, tick_len)], fill=color, width=stroke_px)
        draw.line([(x_left, H - tick_len), (x_left, H)], fill=color, width=stroke_px)

def draw_photo_marks_content(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, img_rects: List[Tuple[int,int,int,int]], color, stroke_px):
    L = mm_to_px(layout.photo_tick_len_mm, pagespec.dpi)
    for (cx, cy, w, h) in img_rects:
        midy = cy + h//2
        draw.line([(cx, midy - L//2), (cx, midy + L//2)], fill=color, width=stroke_px)
        draw.line([(cx + w, midy - L//2), (cx + w, midy + L//2)], fill=color, width=stroke_px)
        midx = cx + w//2
        draw.line([(midx - L//2, cy), (midx + L//2, cy)], fill=color, width=stroke_px)
        draw.line([(midx - L//2, cy + h), (midx + L//2, cy + h)], fill=color, width=stroke_px)

def draw_photo_marks_in_margins(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, img_rects: List[Tuple[int,int,int,int]], color, stroke_px):
    W, H = pagespec.size_px
    L = mm_to_px(layout.photo_tick_len_mm, pagespec.dpi)
    # Per-edge offsets (mm) resolved to page edges based on binding side
    offB = getattr(layout, "photo_margin_mark_offset_b_mm", 0.0)
    offD = getattr(layout, "photo_margin_mark_offset_d_mm", 0.0)
    offU = getattr(layout, "photo_margin_mark_offset_u_mm", 0.0)
    offN = getattr(layout, "photo_margin_mark_offset_n_mm", 0.0)
    bs = getattr(layout, "binding_side", "left")
    # Defaults: U->top, N->bottom, left/right 0 unless binding is left/right
    top_mm = offU
    bottom_mm = offN
    left_mm = 0.0
    right_mm = 0.0
    if bs == "left":
        left_mm, right_mm = offB, offD
    elif bs == "right":
        right_mm, left_mm = offB, offD
    elif bs == "top":
        top_mm, bottom_mm = offB, offD
    elif bs == "bottom":
        bottom_mm, top_mm = offB, offD
    offx_left  = mm_to_px(left_mm, pagespec.dpi)
    offx_right = mm_to_px(right_mm, pagespec.dpi)
    offy_top   = mm_to_px(top_mm, pagespec.dpi)
    offy_bottom= mm_to_px(bottom_mm, pagespec.dpi)
    for (cx, cy, w, h) in img_rects:
        x_left = cx + offx_left; x_right = cx + w + offx_right
        draw.line([(x_left, 0), (x_left, L)], fill=color, width=stroke_px)
        draw.line([(x_left, H - L), (x_left, H)], fill=color, width=stroke_px)
        draw.line([(x_right, 0), (x_right, L)], fill=color, width=stroke_px)
        draw.line([(x_right, H - L), (x_right, H)], fill=color, width=stroke_px)
        y_top = cy + offy_top; y_bottom = cy + h + offy_bottom
        draw.line([(0, y_top), (L, y_top)], fill=color, width=stroke_px)
        draw.line([(W - L, y_top), (W, y_top)], fill=color, width=stroke_px)
        draw.line([(0, y_bottom), (L, y_bottom)], fill=color, width=stroke_px)
        draw.line([(W - L, y_bottom), (W, y_bottom)], fill=color, width=stroke_px)

def draw_text_centered(draw: ImageDraw.ImageDraw, text: str, center_xy: Tuple[int,int], font: ImageFont.FreeTypeFont, color=(0,0,0,255)):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        try:
            tw, th = font.getsize(text)
        except Exception:
            tw, th = (0,0)
    x = center_xy[0] - tw//2
    y = center_xy[1] - th//2
    draw.text((x,y), text, fill=color, font=font)

def draw_binding_overlays(draw: ImageDraw.ImageDraw, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, color, stroke_px):
    # Outline rectangles indicating the binding strip region inside each card
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.card_h_mm, pagespec.dpi)
    bind_px = mm_to_px(layout.binding_margin_mm, pagespec.dpi)
    padL = mm_to_px(layout.pad_left_mm, pagespec.dpi)
    padT = mm_to_px(layout.pad_top_mm, pagespec.dpi)

    for r in range(grid.rows):
        for c in range(grid.cols):
            xc = x0 + c*(cell_w_px + gutter_px)
            yc = y0 + r*(cell_h_px + gutter_px)
            ix0 = xc + padL
            iy0 = yc + padT
            if layout.binding_side == "left":
                # strip is immediately to the left of the image area within the card
                bx0, by0 = xc, yc
                bx1, by1 = xc + bind_px, yc + cell_h_px
            elif layout.binding_side == "right":
                bx0, by0 = xc + cell_w_px - bind_px, yc
                bx1, by1 = xc + cell_w_px, yc + cell_h_px
            elif layout.binding_side == "top":
                bx0, by0 = xc, yc
                bx1, by1 = xc + cell_w_px, yc + bind_px
            else:  # bottom
                bx0, by0 = xc, yc + cell_h_px - bind_px
                bx1, by1 = xc + cell_w_px, yc + cell_h_px
            draw.rectangle([bx0, by0, bx1, by1], outline=color, width=stroke_px)

def compose_pages(frames, pagespec: PageSpec, layout: LayoutSpec, grid: Grid, reverse_order=False, base_filename: str = ""):
    W, H = pagespec.size_px
    x0 = mm_to_px(pagespec.margin_left_mm, pagespec.dpi)
    y0 = mm_to_px(pagespec.margin_top_mm, pagespec.dpi)
    mb_px = mm_to_px(pagespec.margin_bottom_mm, pagespec.dpi)
    gutter_px = mm_to_px(layout.gutter_mm, pagespec.dpi)
    bind_px = mm_to_px(layout.binding_margin_mm, pagespec.dpi)
    padL = mm_to_px(layout.pad_left_mm, pagespec.dpi)
    padR = mm_to_px(layout.pad_right_mm, pagespec.dpi)
    padT = mm_to_px(layout.pad_top_mm, pagespec.dpi)
    padB = mm_to_px(layout.pad_bottom_mm, pagespec.dpi)

    img_w_px = mm_to_px(grid.img_w_mm, pagespec.dpi)
    img_h_px = mm_to_px(grid.img_h_mm, pagespec.dpi)
    cell_w_px = mm_to_px(grid.card_w_mm, pagespec.dpi)
    cell_h_px = mm_to_px(grid.card_h_mm, pagespec.dpi)

    seq = list(reversed(frames)) if reverse_order else list(frames)
    pages = []
    idx = 0

    stroke_px = max(1, mm_to_px(layout.tick_thickness_mm, pagespec.dpi))

    page_index = 0
    frame_counter = 0

    filename_font, _, _ = try_load_font(max(8, mm_to_px(layout.filename_font_mm, pagespec.dpi)), layout.user_font_path)
    number_font, _, _ = try_load_font(max(8, mm_to_px(layout.frame_number_font_mm, pagespec.dpi)), layout.user_font_path)
    debug_font, _, _ = try_load_font(max(6, mm_to_px(layout.debug_font_mm, pagespec.dpi)), layout.user_font_path)

    while idx < len(seq):
        page_index += 1
        page = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        img_rects = []

        # frames
        for r in range(grid.rows):
            for c in range(grid.cols):
                if idx >= len(seq):
                    break
                fx = seq[idx]; idx += 1; frame_counter += 1

                xc = x0 + c*(cell_w_px + gutter_px)
                yc = y0 + r*(cell_h_px + gutter_px)

                inner_x0 = xc + padL
                inner_y0 = yc + padT
                if layout.binding_side == "left":
                    inner_x0 += bind_px
                elif layout.binding_side == "top":
                    inner_y0 += bind_px

                cx = inner_x0; cy = inner_y0
                fr = fx.resize((img_w_px, img_h_px), Image.LANCZOS)
                page.paste(fr, (cx, cy), fr)
                img_rects.append((cx, cy, img_w_px, img_h_px))

                if layout.show_image_debug_overlay:
                    label = f"{grid.img_w_mm:.1f}×{grid.img_h_mm:.1f} mm @ ({px_to_mm(cx, pagespec.dpi):.1f},{px_to_mm(cy, pagespec.dpi):.1f})"
                    draw.text((cx, cy), label, fill=(0,0,0,255), font=debug_font)

                if layout.show_frame_numbers_in_binding and layout.binding_margin_mm > 0:
                    if layout.binding_side == "left":
                        bx0, by0, bx1, by1 = xc, yc, xc + bind_px, yc + cell_h_px
                    elif layout.binding_side == "right":
                        bx0, by0, bx1, by1 = xc + cell_w_px - bind_px, yc, xc + cell_w_px, yc + cell_h_px
                    elif layout.binding_side == "top":
                        bx0, by0, bx1, by1 = xc, yc, xc + cell_w_px, yc + bind_px
                    else:
                        bx0, by0, bx1, by1 = xc, yc + cell_h_px - bind_px, xc + cell_w_px, yc + cell_h_px
                    cxn = (bx0 + bx1)//2; cyn = (by0 + by1)//2
                    num = layout.frame_start_number + frame_counter - 1
                    draw_text_centered(draw, str(num), (cxn, cyn), number_font)

        # overlays
        if layout.show_margin_box:
            draw_margin_box(draw, pagespec, color=layout.color_margin_box, stroke_px=stroke_px)
        if layout.show_card_grid:
            draw_card_grid(draw, pagespec, layout, grid, color=(0,0,0,80), stroke_px=stroke_px)
        if layout.show_gutter_guides:
            draw_gutter_guides(draw, pagespec, layout, grid, color=layout.color_gutter_guides, stroke_px=stroke_px)
        if layout.show_card_cut_marks:
            draw_card_cut_marks(draw, pagespec, layout, grid, color=layout.color_card_marks, stroke_px=stroke_px)
        if layout.show_cell_left_edge_marks_margins:
            draw_cell_left_edge_marks_in_margins(draw, pagespec, layout, grid, color=layout.color_cell_left_marks, stroke_px=stroke_px)
        if layout.show_photo_edge_marks_content and img_rects:
            draw_photo_marks_content(draw, pagespec, layout, img_rects, color=layout.color_photo_marks_content, stroke_px=stroke_px)
        if layout.show_photo_edge_marks_margins and img_rects:
            draw_photo_marks_in_margins(draw, pagespec, layout, img_rects, color=layout.color_photo_marks_margins, stroke_px=stroke_px)
        if layout.show_binding_strip_overlay and layout.binding_margin_mm > 0:
            draw_binding_overlays(draw, pagespec, layout, grid, color=layout.color_binding_overlay, stroke_px=stroke_px)

        if layout.show_filename_in_bottom_margin and base_filename:
            base_y = H - mb_px//2
            x = mm_to_px(pagespec.margin_left_mm, pagespec.dpi) + mm_to_px(layout.filename_x_offset_mm, pagespec.dpi)
            y = base_y - mm_to_px(layout.filename_y_offset_mm, pagespec.dpi)
            text = f"{base_filename} — page {page_index}"
            draw.text((x, y - mm_to_px(layout.filename_font_mm, pagespec.dpi)//2), text, fill=layout.filename_color, font=filename_font)

        pages.append(page)
    return pages

def save_pages_to_pdf(pages: List[Image.Image]) -> bytes:
    if not pages:
        return b""
    buf = io.BytesIO()
    first, rest = pages[0], pages[1:]
    first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()

# ----------------------- UI -----------------------
st.set_page_config(page_title="daft junk flip booker", page_icon="📘", layout="wide")
st.title("daft junk flip booker")
st.caption("Flipbook layout generator for GIFs — precise cut marks, binding strips, and labels.")

# Styled expander headers
st.markdown(
    '''
    <style>
    [data-testid="stExpander"] > details > summary {
        background: #f1f5ff !important;
        border: 1px solid #c9d6ff !important;
        color: #0c2a6e !important;
        border-radius: 10px;
        padding: 8px 12px;
    }
    [data-testid="stExpander"] > details[open] > summary {
        background: #e6eeff !important;
    }
    </style>
    ''',
    unsafe_allow_html=True,
)

# Main uploader first so we know native size before sidebar order
uploaded = st.file_uploader("Upload an animated GIF", type=["gif"])
raw = None
native_w_mm = None; native_h_mm = None; aspect = 1.0
if uploaded:
    try:
        raw = Image.open(uploaded)
        fw, fh = raw.size
        # Only first frame is needed to know native aspect/size
        dpi_preview = 300
        native_w_mm = px_to_mm(fw, dpi_preview)
        native_h_mm = px_to_mm(fh, dpi_preview)
        aspect = native_h_mm / native_w_mm if native_w_mm else 1.0
    except Exception as e:
        st.error(f"Could not open GIF: {e}")

# Sidebar (expanders closed by default). Image & Scaling goes first (when uploaded).
with st.sidebar:
    if uploaded and raw is not None:
        with st.expander("Image & Scaling", expanded=False):
            if "size_state" not in st.session_state:
                st.session_state.size_state = {"mode":"Scale %", "scale":100.0, "w":native_w_mm, "h":native_h_mm, "lock":True, "prev_w":native_w_mm, "prev_h":native_h_mm}
            mode = st.radio("Sizing mode", ["Scale %", "Locked (🔒)", "W×H"], index=["Scale %","Locked (🔒)","W×H"].index(st.session_state.size_state.get("mode","Scale %")), horizontal=True, key="mode_radio")
            st.session_state.size_state["mode"] = mode

            if mode == "Scale %":
                s = st.slider("Scale vs native (%)", 5, 800, int(st.session_state.size_state.get("scale", 100.0)), step=1)
                scale_percent = float(st.number_input("Scale precise (%)", 5.0, 800.0, float(s), 0.1, format="%.1f"))
                st.session_state.size_state["scale"] = scale_percent
                target_w_mm = None; target_h_mm = None
                size_mode = "scale"
                show_w = native_w_mm * (scale_percent/100.0)
                show_h = native_h_mm * (scale_percent/100.0)

            elif mode == "Locked (🔒)":
                lock = st.checkbox("🔒 Lock aspect", value=st.session_state.size_state.get("lock", True))
                st.session_state.size_state["lock"] = lock
                c1, c2 = st.columns(2)
                with c1:
                    w_in = st.number_input("Width (mm)", 5.0, 1000.0, float(st.session_state.size_state.get("w", native_w_mm)), 0.1, key="locked_w")
                with c2:
                    h_in = st.number_input("Height (mm)", 5.0, 1000.0, float(st.session_state.size_state.get("h", native_h_mm)), 0.1, key="locked_h")
                prev_w = st.session_state.size_state.get("prev_w", w_in)
                prev_h = st.session_state.size_state.get("prev_h", h_in)
                changed_w = abs(w_in - prev_w) > 1e-6
                changed_h = abs(h_in - prev_h) > 1e-6
                if lock:
                    if changed_w and not changed_h:
                        h_in = w_in * aspect
                    elif changed_h and not changed_w:
                        w_in = h_in / aspect if aspect else h_in
                st.session_state.size_state["w"] = w_in
                st.session_state.size_state["h"] = h_in
                st.session_state.size_state["prev_w"] = w_in
                st.session_state.size_state["prev_h"] = h_in
                size_mode = "locked"
                target_w_mm = w_in if changed_w or lock else None
                target_h_mm = h_in if changed_h or lock else None
                show_w, show_h = w_in, h_in

            else:  # W×H
                c1, c2 = st.columns(2)
                with c1:
                    w2 = st.number_input("Width (mm)", 5.0, 1000.0, float(st.session_state.size_state.get("w", native_w_mm)), 0.1, key="ind_w")
                with c2:
                    h2 = st.number_input("Height (mm)", 5.0, 1000.0, float(st.session_state.size_state.get("h", native_h_mm)), 0.1, key="ind_h")
                st.session_state.size_state["w"] = w2
                st.session_state.size_state["h"] = h2
                size_mode = "independent"
                target_w_mm = w2; target_h_mm = h2
                show_w, show_h = w2, h2

            reset = st.button("Reset to native (100%)")
            if reset:
                st.session_state.size_state = {"mode":mode, "scale": 100.0, "w": native_w_mm, "h": native_h_mm, "lock": True, "prev_w": native_w_mm, "prev_h": native_h_mm}
                st.rerun()

            st.markdown(f"**Live image size:** {show_w:.2f} × {show_h:.2f} mm")
    else:
        with st.expander("Image & Scaling", expanded=False):
            st.info("Upload a GIF to unlock scaling controls.")

    with st.expander("Page", expanded=False):
        dpi = st.slider("DPI", 150, 600, 300, step=25)
        size_choice = st.selectbox("Paper size", list(PAPER_SIZES_MM.keys()), index=0)
        if size_choice == "Custom":
            page_w_mm = st.number_input("Custom width (mm)", 50.0, 1000.0, 210.0, 0.1)
            page_h_mm = st.number_input("Custom height (mm)", 50.0, 1000.0, 297.0, 0.1)
        else:
            page_w_mm, page_h_mm = PAPER_SIZES_MM[size_choice]
        orient = st.radio("Orientation", ["Portrait", "Landscape"], index=0, horizontal=True)
        if orient == "Landscape":
            page_w_mm, page_h_mm = page_h_mm, page_w_mm

        st.subheader("Margins (mm)")
        cA, cB = st.columns(2)
        with cA:
            margin_top_mm = st.number_input("Top", 0.0, 40.0, 8.0, 0.1)
            margin_left_mm = st.number_input("Left", 0.0, 40.0, 8.0, 0.1)
        with cB:
            margin_right_mm = st.number_input("Right", 0.0, 40.0, 8.0, 0.1)
            margin_bottom_mm = st.number_input("Bottom", 0.0, 40.0, 8.0, 0.1)

    with st.expander("Grid & Cutting", expanded=False):
        gutter_mm_slider = st.slider("Gutter / cut channel (mm) [slider]", 0.0, 40.0, 2.0, 0.5)
        binding_mm_slider = st.slider("Binding strip width (mm) [slider]", 0.0, 40.0, 6.0, 0.5)
        gutter_mm = st.number_input("Gutter (mm) precise", 0.0, 100.0, gutter_mm_slider, 0.1)
        binding_mm = st.number_input("Binding width (mm) precise", 0.0, 100.0, binding_mm_slider, 0.1)

        flip_from = st.radio("Flip from", ["Right", "Left", "Top", "Bottom"], index=0, horizontal=True)
        flip_from = flip_from.lower()
        opposite = {"right":"left","left":"right","top":"bottom","bottom":"top"}
        binding_side = opposite[flip_from]
        st.info(f"Binding strip on **{binding_side.upper()}** (opposite of flip).", icon="📎")

    with st.expander("Placement", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            pad_left = st.number_input("Left padding (mm)", 0.0, 100.0, 0.0, 0.1)
            pad_top = st.number_input("Top padding (mm)", 0.0, 100.0, 0.0, 0.1)
        with c2:
            pad_right = st.number_input("Right padding (mm)", 0.0, 100.0, 0.0, 0.1)
            pad_bottom = st.number_input("Bottom padding (mm)", 0.0, 100.0, 0.0, 0.1)

    with st.expander("Overlays", expanded=False):
        st.markdown("**Toggles**")
        show_margin_box = st.checkbox("Margin box", value=True)
        show_card_grid = st.checkbox("Card grid (cell rectangles)", value=False)
        show_gutter_guides = st.checkbox("Gutter guides (full-length)", value=True)
        show_card_cut_marks = st.checkbox("Page-edge cut marks at cell boundaries", value=True)
        show_cell_left_edge_marks_margins = st.checkbox("Page-edge cut marks at each cell LEFT edge", value=False)
        show_photo_edge_marks_content = st.checkbox("Photo-edge cut marks (on image)", value=False)
        show_photo_edge_marks_margins = st.checkbox("Photo-edge cut marks (in margins)", value=False)
        show_binding_strip_overlay = st.checkbox("Show binding strip overlay (outline)", value=False)

        st.markdown("---")
        st.markdown("**Colors & geometry**")
        margin_box_color = st.color_picker("Margin box color", "#000000")
        color_gutter_guides = st.color_picker("Gutter guide color", "#4B89DC")
        color_card_marks = st.color_picker("Card boundary marks (page-edge)", "#000000")
        color_cell_left = st.color_picker("Cell LEFT-edge marks (margins)", "#FF0000")
        color_photo_marks_content = st.color_picker("Photo-edge marks (on image)", "#00AA00")
        color_photo_marks_margins = st.color_picker("Photo-edge marks (margins)", "#AA00AA")
        color_binding_overlay = st.color_picker("Binding strip overlay color", "#FF8800")
        tick_len_mm = st.number_input("Card/cell tick length (mm)", 1.0, 30.0, 3.0, 0.1)
        photo_tick_len_mm = st.number_input("Photo-edge tick length (mm)", 1.0, 30.0, 3.0, 0.1)
        tick_thickness_mm = st.number_input("Cut mark thickness (mm)", 0.1, 3.0, 0.2, 0.1)

    with st.expander("Labels", expanded=False):
        show_frame_numbers_in_binding = st.checkbox("Number each frame in binding strip", value=True)
        frame_start_number = st.number_input("Start number", 1, 999999, 1, step=1)
        frame_number_font_mm = st.number_input("Frame number font size (mm)", 1.0, 30.0, 2.0, 0.1)
        frame_number_near_spine = st.checkbox("Place frame number near spine edge", value=True)
        frame_number_rotate_vertical = st.checkbox("Rotate numbers for top/bottom binding (not used yet)", value=False)
        st.markdown("---")
        show_filename_in_bottom_margin = st.checkbox("Show filename at bottom margin (with page number)", value=True)
        filename_font_mm = st.number_input("Filename font size (mm)", 1.0, 30.0, 2.0, 0.1)
        filename_color_hex = st.color_picker("Filename color", "#000000")
        filename_x_offset_mm = st.number_input("Filename X offset from LEFT page margin (mm)", -200.0, 200.0, 0.0, 0.1)
        filename_y_offset_mm = st.number_input("Filename Y offset within bottom margin (mm)", -50.0, 50.0, 0.0, 0.1)

    with st.expander("Debug & Fonts", expanded=False):
        show_image_debug_overlay = st.checkbox("Show image size & (x,y) overlay", value=False)
        debug_font_mm = st.number_input("Debug overlay font size (mm)", 1.0, 20.0, 2.0, 0.1)
        user_font = st.file_uploader("Upload a .ttf font (optional)", type=["ttf"])
        user_font_path = None
        if user_font is not None:
            user_font_path = "_user_font.ttf"
            with open(user_font_path, "wb") as f:
                f.write(user_font.read())

    with st.expander("Frames", expanded=False):
        step = st.number_input("Use every Nth frame", 1, 20, 1)
        max_frames = st.number_input("Max frames (0 = all)", 0, 500, 0)
        reverse_order = st.checkbox("Reverse order", value=False)

if not uploaded or raw is None:
    st.info("Upload a GIF to begin.")
else:
    st.success("GIF loaded.")
    # Now that sidebar choices exist, build specs and extract frames
    pagespec = PageSpec(
        width_mm=page_w_mm, height_mm=page_h_mm, dpi=dpi,
        margin_top_mm=margin_top_mm, margin_right_mm=margin_right_mm,
        margin_bottom_mm=margin_bottom_mm, margin_left_mm=margin_left_mm
    )

    frames = extract_frames(raw, step=step, max_frames=(None if max_frames == 0 else max_frames))
    if not frames:
        st.warning("No frames extracted with the current settings.")
        st.stop()

    st.write(f"Extracted **{len(frames)}** frame(s)")

    # Use the actual DPI-selected native for size math
    fw, fh = frames[0].size
    # Recompute native at chosen dpi for size math
    native_w_mm = px_to_mm(fw, pagespec.dpi)
    native_h_mm = px_to_mm(fh, pagespec.dpi)
    aspect = native_h_mm / native_w_mm if native_w_mm else 1.0

    # Size mode determined in sidebar scaling expander
    size_state = st.session_state.get("size_state", {"mode":"Scale %","scale":100.0,"w":native_w_mm,"h":native_h_mm,"lock":True})
    mode = size_state.get("mode", "Scale %")
    if mode == "Scale %":
        size_mode = "scale"
        target_w_mm = None; target_h_mm = None
        scale_percent = float(size_state.get("scale", 100.0))
    elif mode == "Locked (🔒)":
        size_mode = "locked"
        target_w_mm = float(size_state.get("w", native_w_mm))
        target_h_mm = float(size_state.get("h", native_h_mm))
        scale_percent = None
    else:
        size_mode = "independent"
        target_w_mm = float(size_state.get("w", native_w_mm))
        target_h_mm = float(size_state.get("h", native_h_mm))
        scale_percent = None

    
    # --- Safe defaults for per-side photo-edge offsets (ensure defined even if UI not run yet) ---
    photo_margin_mark_offset_b_mm = locals().get("photo_margin_mark_offset_b_mm", 0.0)
    photo_margin_mark_offset_d_mm = locals().get("photo_margin_mark_offset_d_mm", 0.0)
    photo_margin_mark_offset_u_mm = locals().get("photo_margin_mark_offset_u_mm", 0.0)
    photo_margin_mark_offset_n_mm = locals().get("photo_margin_mark_offset_n_mm", 0.0)
    layout = LayoutSpec(
        gutter_mm=gutter_mm, binding_margin_mm=binding_mm,
        flip_from=flip_from, binding_side=binding_side,
        size_mode=size_mode, target_w_mm=target_w_mm, target_h_mm=target_h_mm, scale_percent=scale_percent,
        pad_left_mm=pad_left, pad_right_mm=pad_right, pad_top_mm=pad_top, pad_bottom_mm=pad_bottom,
        show_margin_box=show_margin_box, show_card_grid=show_card_grid,
        show_gutter_guides=show_gutter_guides, show_card_cut_marks=show_card_cut_marks,
        show_cell_left_edge_marks_margins=show_cell_left_edge_marks_margins,
        show_binding_strip_overlay=show_binding_strip_overlay,
        show_photo_edge_marks_content=show_photo_edge_marks_content,
        show_photo_edge_marks_margins=show_photo_edge_marks_margins,
        tick_len_mm=tick_len_mm, photo_tick_len_mm=photo_tick_len_mm, tick_thickness_mm=tick_thickness_mm,
        color_card_marks=hex_to_rgba(color_card_marks),
        color_cell_left_marks=hex_to_rgba(color_cell_left),
        color_gutter_guides=hex_to_rgba(color_gutter_guides),
        color_photo_marks_content=hex_to_rgba(color_photo_marks_content),
        color_photo_marks_margins=hex_to_rgba(color_photo_marks_margins),
        color_margin_box=hex_to_rgba(margin_box_color, 160),
        color_binding_overlay=hex_to_rgba(color_binding_overlay),
        show_frame_numbers_in_binding=show_frame_numbers_in_binding, frame_start_number=frame_start_number,
        frame_number_font_mm=frame_number_font_mm, frame_number_near_spine=frame_number_near_spine,
        frame_number_rotate_vertical=frame_number_rotate_vertical,
        show_filename_in_bottom_margin=show_filename_in_bottom_margin,
        filename_font_mm=filename_font_mm, filename_color=hex_to_rgba(filename_color_hex),
        filename_x_offset_mm=filename_x_offset_mm, filename_y_offset_mm=filename_y_offset_mm,
        photo_margin_mark_offset_b_mm=photo_margin_mark_offset_b_mm,
        photo_margin_mark_offset_d_mm=photo_margin_mark_offset_d_mm,
        photo_margin_mark_offset_u_mm=photo_margin_mark_offset_u_mm,
        photo_margin_mark_offset_n_mm=photo_margin_mark_offset_n_mm,
        show_image_debug_overlay=show_image_debug_overlay, debug_font_mm=debug_font_mm,
        user_font_path=(None),
    )

    img_w_mm, img_h_mm = compute_image_size_mm(fw, fh, pagespec, layout)
    card_w_mm, card_h_mm = compute_card_size_mm(img_w_mm, img_h_mm, layout)
    grid = compute_grid(pagespec, layout, card_w_mm, card_h_mm, img_w_mm, img_h_mm)

    if grid.cols == 0 or grid.rows == 0:
        st.error("Image size + margins/gutter/binding/padding do not fit on the selected paper at this DPI. Reduce size or margins, or change orientation/paper.")
        st.stop()

    st.subheader("Layout summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cards per page", grid.per_page)
        st.write(f"{grid.cols} × {grid.rows}")
    with col2:
        st.write(f"Image: **{img_w_mm:.1f} × {img_h_mm:.1f} mm**")
        st.write(f"Binding: **{binding_mm:.1f} mm** on **{binding_side}** (flip from {flip_from})")
    with col3:
        Wpx, Hpx = pagespec.size_px
        st.write(f"{size_choice} @ {dpi} DPI → **{Wpx} × {Hpx} px**")
        st.write(f"Margins T/R/B/L: **{margin_top_mm:.1f}/{margin_right_mm:.1f}/{margin_bottom_mm:.1f}/{margin_left_mm:.1f} mm**")

    frames_list = frames  # to avoid confusion with PIL frames iterator

    pages = compose_pages(frames_list, pagespec, layout, grid, reverse_order=reverse_order, base_filename=uploaded.name)

    st.subheader("Preview (all pages)")
    st.image(pages, caption=[f"Page {i+1}" for i in range(len(pages))], use_column_width=True)

    pdf = save_pages_to_pdf(pages)
    st.download_button("⬇️ Download PDF", pdf, "flipbook_app_v9_4_output.pdf", "application/pdf")
