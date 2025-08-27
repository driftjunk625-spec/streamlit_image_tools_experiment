"""
Further Flip — v0.2 (2025-08-27)

Patch: enforce bleed < gutter/2 (strict) instead of ≤.
- In sidebar, bleed is clamped if user enters too large.
- Gutter auto-adjust optionally increases to ≥ 2× bleed.
- Rest unchanged from v0.1.
"""

# This snippet must run *after* the sidebar UI defines enable_full_bleed, bleed_mm_val, gutter_mm, auto_gutter_for_bleed, and safe_area_mm_val.
# Make sure to paste this block right before creating LayoutSpec(...)

# --- Enforce strict full-bleed constraint (bleed < gutter/2) and optional auto-gutter ---
if enable_full_bleed and bleed_mm_val > 0:
    desired_bleed = float(bleed_mm_val)
    half_gutter = float(gutter_mm) / 2.0
    if desired_bleed >= half_gutter:
        # strictly less than half gutter
        bleed_mm_effective = max(0.0, half_gutter - 1e-6)
        st.warning(f"Bleed reduced to {bleed_mm_effective:.2f} mm so that gutter > 2× bleed.", icon="⚠️")
    else:
        bleed_mm_effective = desired_bleed
    gutter_mm_effective = max(float(gutter_mm), 2.0 * bleed_mm_effective) if auto_gutter_for_bleed else float(gutter_mm)
else:
    bleed_mm_effective = float(bleed_mm_val)
    gutter_mm_effective = float(gutter_mm)

layout = LayoutSpec(
    gutter_mm=gutter_mm_effective, binding_margin_mm=binding_mm,
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
    show_image_debug_overlay=show_image_debug_overlay, debug_font_mm=debug_font_mm,
    user_font_path=(None),
    enable_full_bleed=enable_full_bleed,
    bleed_mm=bleed_mm_effective,
    safe_area_mm=float(safe_area_mm_val),
    show_trim_box=bool(show_trim_box),
    show_bleed_box=bool(show_bleed_box),
    show_safe_box=bool(show_safe_box),
    color_trim_box=hex_to_rgba(color_trim_hex),
    color_bleed_box=hex_to_rgba(color_bleed_hex, 180),
    color_safe_box=hex_to_rgba(color_safe_hex, 180),
)
