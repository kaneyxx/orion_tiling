"""
Multiplex Patch Visualizer

A Gradio-based tool to compare biomarker and H&E patches at the same coordinates.
Supports browsing through subjects (CRC01, CRC02, etc.) and viewing all 14 image
channels (13 biomarkers + H&E) for any given coordinate.

Note: CLAHE enhancement is applied during extraction (patch_extraction_nolabel.py),
not in this visualizer.

Usage:
    pip install gradio  # if not installed
    python app.py
    # Open http://localhost:7860 in browser
"""

import os
import glob
import numpy as np
import h5py
import gradio as gr
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple

# ============== Configuration ==============
DATASET_DIR = "/home/fas994/o2_scratch"

# Thumbnail settings
THUMB_SIZE = 16        # Each tile shrinks to 16x16 pixels
TILE_SIZE = 1024       # Original tile size

# Biomarkers in display order
BIOMARKERS = [
    "CD45", "CD31", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "PD-1", "Ki67"
]

# All channels including HE
ALL_CHANNELS = ["HE"] + BIOMARKERS


# ============== Helper Functions ==============

def get_subjects() -> List[str]:
    """Get list of available subjects from dataset directory."""
    if not os.path.exists(DATASET_DIR):
        return []

    subjects = []
    for name in sorted(os.listdir(DATASET_DIR)):
        subject_path = os.path.join(DATASET_DIR, name)
        if os.path.isdir(subject_path) and name.startswith("CRC"):
            # Check if tiles.h5 exists (nolabel format)
            h5_path = os.path.join(subject_path, "tiles.h5")
            if os.path.exists(h5_path):
                subjects.append(name)

    return subjects


def get_coordinates(subject: str) -> List[str]:
    """
    Extract all unique coordinates from HE directory.

    Parses filenames like: CRC01_HE_10320_10664.png
    Returns coordinates like: "10320_10664"
    """
    if not subject:
        return []

    he_dir = os.path.join(DATASET_DIR, subject, "HE")
    if not os.path.exists(he_dir):
        return []

    coords = []
    for filename in os.listdir(he_dir):
        if filename.endswith(".png"):
            # Parse: CRC01_HE_10320_10664.png
            parts = filename.replace(".png", "").split("_")
            if len(parts) >= 4:
                x, y = parts[-2], parts[-1]
                coords.append(f"{x}_{y}")

    # Sort by x, then y coordinate
    coords = sorted(set(coords), key=lambda c: (int(c.split("_")[0]), int(c.split("_")[1])))
    return coords


def load_image(path: str) -> Optional[Image.Image]:
    """Load an image from path, return None if not found."""
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    return None


def find_biomarker_image(subject: str, biomarker: str, coordinate: str) -> Optional[str]:
    """
    Find biomarker image path (handles pos/neg in filename).

    Pattern: CRC01_CD45_neg_10320_10664.png or CRC01_CD45_pos_10320_10664.png
    """
    bm_dir = os.path.join(DATASET_DIR, subject, biomarker)
    if not os.path.exists(bm_dir):
        return None

    # Try both pos and neg patterns
    for label in ["pos", "neg"]:
        path = os.path.join(bm_dir, f"{subject}_{biomarker}_{label}_{coordinate}.png")
        if os.path.exists(path):
            return path

    return None


def has_hdf5(subject: str) -> bool:
    """Check if subject has HDF5 format biomarkers."""
    h5_path = os.path.join(DATASET_DIR, subject, "tiles.h5")
    return os.path.exists(h5_path)


def load_biomarker_from_hdf5(subject: str, biomarker: str, coordinate: str) -> Optional[Image.Image]:
    """
    Load biomarker image from HDF5 file.
    Reconstructs RGB from R (biomarker), G (Pan-CK), B (Hoechst).
    """
    h5_path = os.path.join(DATASET_DIR, subject, "tiles.h5")
    if not os.path.exists(h5_path):
        return None

    try:
        with h5py.File(h5_path, 'r') as f:
            # Get coordinate index
            coords = [c.decode() if isinstance(c, bytes) else c for c in f['coordinates'][:]]
            if coordinate not in coords:
                return None

            idx = coords.index(coordinate)

            # Load channels
            r = f[f'{biomarker}_R'][idx]
            g = f['shared_G'][idx]
            b = f['shared_B'][idx]

            # Stack to RGB
            rgb = np.stack([r, g, b], axis=-1)
            return Image.fromarray(rgb)
    except Exception as e:
        print(f"Error loading from HDF5: {e}")
        return None


def load_all_images(subject: str, coordinate: str) -> Dict[str, Optional[Image.Image]]:
    """Load all 14 images for a given subject and coordinate."""
    images = {}

    if not subject or not coordinate:
        return {ch: None for ch in ALL_CHANNELS}

    # Load H&E (always PNG)
    he_path = os.path.join(DATASET_DIR, subject, "HE", f"{subject}_HE_{coordinate}.png")
    images["HE"] = load_image(he_path)

    # Load biomarkers - check if HDF5 exists
    use_hdf5 = has_hdf5(subject)

    if use_hdf5:
        # Load from HDF5
        for bm in BIOMARKERS:
            images[bm] = load_biomarker_from_hdf5(subject, bm, coordinate)
    else:
        # Load from PNG
        for bm in BIOMARKERS:
            bm_path = find_biomarker_image(subject, bm, coordinate)
            images[bm] = load_image(bm_path) if bm_path else None

    return images


# ============== Thumbnail Functions ==============

def get_thumbnail_path(subject: str) -> str:
    """Get path to thumbnail image for a subject (stored with dataset)."""
    return os.path.join(DATASET_DIR, subject, f"{subject}_HE_thumbnail.png")


def generate_thumbnail(subject: str) -> Optional[Image.Image]:
    """Generate or load cached thumbnail for subject."""
    thumb_path = get_thumbnail_path(subject)

    # Return cached if exists
    if os.path.exists(thumb_path):
        return Image.open(thumb_path)

    # Generate new thumbnail
    he_dir = os.path.join(DATASET_DIR, subject, "HE")
    if not os.path.exists(he_dir):
        return None

    # Collect coordinates and find grid size
    coords = []
    for fn in os.listdir(he_dir):
        if fn.endswith('.png'):
            parts = fn.replace('.png', '').split('_')
            if len(parts) >= 4:
                x, y = int(parts[-2]), int(parts[-1])
                coords.append((x, y, fn))

    if not coords:
        return None

    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)

    out_width = (max_x // TILE_SIZE + 1) * THUMB_SIZE
    out_height = (max_y // TILE_SIZE + 1) * THUMB_SIZE

    output = Image.new('RGB', (out_width, out_height), (255, 255, 255))

    for x, y, fn in coords:
        img = Image.open(os.path.join(he_dir, fn))
        thumb = img.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        out_x = (x // TILE_SIZE) * THUMB_SIZE
        out_y = (y // TILE_SIZE) * THUMB_SIZE
        output.paste(thumb, (out_x, out_y))

    # Cache it (saved alongside dataset)
    output.save(thumb_path)

    return output


def draw_highlight_on_thumbnail(subject: str, coordinate: str) -> Optional[Image.Image]:
    """Draw a red rectangle on thumbnail showing current tile position."""
    thumb = generate_thumbnail(subject)
    if thumb is None:
        return None

    thumb = thumb.copy()  # Don't modify cached version

    # Parse coordinate
    parts = coordinate.split('_')
    if len(parts) != 2:
        return thumb

    tile_x, tile_y = int(parts[0]), int(parts[1])

    # Convert to thumbnail pixel coordinates
    px = (tile_x // TILE_SIZE) * THUMB_SIZE
    py = (tile_y // TILE_SIZE) * THUMB_SIZE

    # Draw red rectangle
    draw = ImageDraw.Draw(thumb)
    draw.rectangle(
        [px, py, px + THUMB_SIZE - 1, py + THUMB_SIZE - 1],
        outline='red',
        width=2
    )

    return thumb


# ============== Gradio Event Handlers ==============

def on_subject_change(subject: str) -> Tuple:
    """When subject changes, update coordinate dropdown and thumbnail."""
    coords = get_coordinates(subject)
    if coords:
        # Generate thumbnail with highlight on first coordinate
        thumb = draw_highlight_on_thumbnail(subject, coords[0])
        # Return updated dropdown choices and select first coordinate
        return (
            gr.update(choices=coords, value=coords[0]),
            coords,
            0,
            f"1 / {len(coords)}",
            thumb
        )
    else:
        return (
            gr.update(choices=[], value=None),
            [],
            0,
            "0 / 0",
            None
        )


def on_coordinate_change(subject: str, coordinate: str, coords: List[str]) -> Tuple:
    """When coordinate changes, load and display all images."""
    images = load_all_images(subject, coordinate)

    # Get current index
    idx = coords.index(coordinate) if coordinate in coords else 0
    counter = f"{idx + 1} / {len(coords)}" if coords else "0 / 0"

    # Update thumbnail with highlight
    thumb = draw_highlight_on_thumbnail(subject, coordinate) if subject and coordinate else None

    # Return images in order + index + counter + thumbnail
    return tuple([images.get(ch) for ch in ALL_CHANNELS]) + (idx, counter, thumb)


def on_prev_click(subject: str, coords: List[str], current_idx: int) -> Tuple:
    """Navigate to previous coordinate."""
    if not coords:
        return (None,) * len(ALL_CHANNELS) + (gr.update(), 0, "0 / 0", None)

    new_idx = (current_idx - 1) % len(coords)
    new_coord = coords[new_idx]
    images = load_all_images(subject, new_coord)
    counter = f"{new_idx + 1} / {len(coords)}"

    # Update thumbnail with highlight
    thumb = draw_highlight_on_thumbnail(subject, new_coord)

    return tuple([images.get(ch) for ch in ALL_CHANNELS]) + (gr.update(value=new_coord), new_idx, counter, thumb)


def on_next_click(subject: str, coords: List[str], current_idx: int) -> Tuple:
    """Navigate to next coordinate."""
    if not coords:
        return (None,) * len(ALL_CHANNELS) + (gr.update(), 0, "0 / 0", None)

    new_idx = (current_idx + 1) % len(coords)
    new_coord = coords[new_idx]
    images = load_all_images(subject, new_coord)
    counter = f"{new_idx + 1} / {len(coords)}"

    # Update thumbnail with highlight
    thumb = draw_highlight_on_thumbnail(subject, new_coord)

    return tuple([images.get(ch) for ch in ALL_CHANNELS]) + (gr.update(value=new_coord), new_idx, counter, thumb)


def on_thumbnail_click(subject: str, coords: List[str], evt: gr.SelectData) -> Tuple:
    """Handle click on thumbnail to jump to that region."""
    if not subject or not coords:
        return (None,) * len(ALL_CHANNELS) + (gr.update(), 0, "0 / 0", None)

    # Get click coordinates
    click_x, click_y = evt.index[0], evt.index[1]

    # Convert to tile coordinates
    tile_x = (click_x // THUMB_SIZE) * TILE_SIZE
    tile_y = (click_y // THUMB_SIZE) * TILE_SIZE
    target_coord = f"{tile_x}_{tile_y}"

    # Find closest valid coordinate if target not in list
    if target_coord not in coords:
        # Find nearest coordinate
        min_dist = float('inf')
        for c in coords:
            cx, cy = map(int, c.split('_'))
            dist = abs(cx - tile_x) + abs(cy - tile_y)
            if dist < min_dist:
                min_dist = dist
                target_coord = c

    # Load images and update
    idx = coords.index(target_coord)
    images = load_all_images(subject, target_coord)

    counter = f"{idx + 1} / {len(coords)}"

    # Update thumbnail with highlight
    thumb_with_highlight = draw_highlight_on_thumbnail(subject, target_coord)

    return tuple([images.get(ch) for ch in ALL_CHANNELS]) + (
        gr.update(value=target_coord), idx, counter, thumb_with_highlight
    )


# ============== Build Gradio Interface ==============

def build_interface():
    """Build and return the Gradio interface."""

    subjects = get_subjects()
    initial_subject = subjects[0] if subjects else None
    initial_coords = get_coordinates(initial_subject) if initial_subject else []
    initial_coord = initial_coords[0] if initial_coords else None

    with gr.Blocks(title="Multiplex Patch Visualizer") as demo:

        gr.Markdown("# Multiplex Patch Visualizer")
        gr.Markdown("Compare H&E and biomarker patches at the same coordinates.")

        # State variables
        coords_state = gr.State(initial_coords)
        idx_state = gr.State(0)

        # Controls row
        with gr.Row():
            subject_dd = gr.Dropdown(
                label="Subject",
                choices=subjects,
                value=initial_subject,
                scale=1
            )
            coord_dd = gr.Dropdown(
                label="Coordinate",
                choices=initial_coords,
                value=initial_coord,
                allow_custom_value=True,
                scale=2
            )
            counter_text = gr.Textbox(
                label="Position",
                value=f"1 / {len(initial_coords)}" if initial_coords else "0 / 0",
                interactive=False,
                scale=1
            )

        # Thumbnail for navigation (click to select tile)
        thumbnail_img = gr.Image(
            label="Overview (click to navigate)",
            interactive=True,
            height=500
        )

        # Image grid - Row 1: HE + first 6 biomarkers
        gr.Markdown("### H&E and Biomarkers")
        with gr.Row():
            img_he = gr.Image(label="H&E", height=200)
            img_cd45 = gr.Image(label="CD45", height=200)
            img_cd31 = gr.Image(label="CD31", height=200)
            img_cd68 = gr.Image(label="CD68", height=200)
            img_cd4 = gr.Image(label="CD4", height=200)
            img_foxp3 = gr.Image(label="FOXP3", height=200)
            img_cd8a = gr.Image(label="CD8a", height=200)

        # Image grid - Row 2: remaining 6 biomarkers
        with gr.Row():
            img_cd45ro = gr.Image(label="CD45RO", height=200)
            img_cd20 = gr.Image(label="CD20", height=200)
            img_pdl1 = gr.Image(label="PD-L1", height=200)
            img_cd3e = gr.Image(label="CD3e", height=200)
            img_cd163 = gr.Image(label="CD163", height=200)
            img_pd1 = gr.Image(label="PD-1", height=200)
            img_ki67 = gr.Image(label="Ki67", height=200)

        # All image components in order matching ALL_CHANNELS
        all_images = [
            img_he, img_cd45, img_cd31, img_cd68, img_cd4, img_foxp3, img_cd8a,
            img_cd45ro, img_cd20, img_pdl1, img_cd3e, img_cd163, img_pd1, img_ki67
        ]

        # Event handlers
        subject_dd.change(
            fn=on_subject_change,
            inputs=[subject_dd],
            outputs=[coord_dd, coords_state, idx_state, counter_text, thumbnail_img]
        )

        coord_dd.change(
            fn=on_coordinate_change,
            inputs=[subject_dd, coord_dd, coords_state],
            outputs=all_images + [idx_state, counter_text, thumbnail_img]
        )

        # Thumbnail click event
        thumbnail_img.select(
            fn=on_thumbnail_click,
            inputs=[subject_dd, coords_state],
            outputs=all_images + [coord_dd, idx_state, counter_text, thumbnail_img]
        )

        # Load initial images and thumbnail
        if initial_subject and initial_coord:
            demo.load(
                fn=lambda s, c, cs: on_coordinate_change(s, c, cs),
                inputs=[subject_dd, coord_dd, coords_state],
                outputs=all_images + [idx_state, counter_text, thumbnail_img]
            )

    return demo


# ============== Main ==============

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=True,
        show_error=True
    )
