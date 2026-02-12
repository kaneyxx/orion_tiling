"""
Label-Free Non-Overlapping Tile Extraction Script (Gating-Based)

Extracts non-overlapping 1024x1024 tiles from multiplex TIFF images WITHOUT using
label CSV files. Uses gating-based normalization (log transform).

Key differences from patch_extraction_gating.py:
- Coordinate Source: Generated grid instead of Label CSV
- Stride: 1024 (non-overlapping) instead of 344 (overlapping)
- Labels: None (label-free extraction)
- Patches (CRC01): ~4,180 instead of ~16,642

HDF5 Structure:
    tiles.h5
    ├── coordinates          # (N,) string - "x_y" format
    ├── shared_G             # (N, 1024, 1024) uint8 - Pan-CK
    ├── shared_B             # (N, 1024, 1024) uint8 - Hoechst
    ├── {marker}_R           # (N, 1024, 1024) uint8
    └── attributes:
        ├── subject
        ├── n_tiles
        ├── patch_size
        ├── stride
        ├── image_height, image_width
        ├── biomarkers
        └── normalization_method: 'gating'

Usage:
    python patch_extraction_nolabel.py --subjects CRC01 --output-format hdf5
    python patch_extraction_nolabel.py --subjects CRC01 --markers PD-L1 CD45
    python patch_extraction_nolabel.py --subjects CRC01 CRC02 CRC03

    # Extract H&E patches at the same coordinates
    python patch_extraction_nolabel.py --subjects CRC01 --mode he
    python patch_extraction_nolabel.py --subjects CRC01 --mode both
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import tifffile
import h5py
import cv2
from PIL import Image
from tqdm import tqdm


# ============== Configuration ==============

# CLAHE default parameters
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)

# Empty tile filtering default parameters
DEFAULT_WHITE_THRESHOLD = 240
DEFAULT_WHITE_RATIO = 0.9
# Channel list - adjust according to your multiplex panel
CH_LI = [
    'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4',
    'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
    'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
]

# Create channel index mapping
CH_MAP_DICT = {k: CH_LI[k] for k in range(len(CH_LI))}
CH_MAP_RES_DICT = {v: k for k, v in CH_MAP_DICT.items()}

# Target columns (immune markers to process)
TARGET_COL = [
    'CD45', 'CD31', 'CD68', 'CD4', 'FOXP3', 'CD8a',
    'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'PD-1', 'Ki67'
]

# Default settings
DEFAULT_PATCH_SIZE = 1024
DEFAULT_GATING_DIR = './gating'


def apply_clahe(img: np.ndarray, clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
                tile_grid_size: tuple = DEFAULT_CLAHE_TILE_GRID_SIZE) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Args:
        img: Input image array (grayscale or RGB, uint8)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid for histogram equalization (default: (8, 8))

    Returns:
        CLAHE-enhanced image array
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img.shape) == 3:
        # Apply CLAHE to each channel separately
        channels = [clahe.apply(img[:, :, i]) for i in range(img.shape[2])]
        return np.stack(channels, axis=-1)
    return clahe.apply(img)


def is_tile_empty(tile: np.ndarray, white_threshold: int = DEFAULT_WHITE_THRESHOLD,
                  white_ratio: float = DEFAULT_WHITE_RATIO) -> bool:
    """
    Check if a tile is mostly empty (white background).

    Args:
        tile: RGB image array
        white_threshold: Pixel value threshold to consider as "white" (default: 240)
        white_ratio: Ratio of white pixels to consider tile as empty (default: 0.9)

    Returns:
        True if tile is considered empty (mostly white background)
    """
    # Check if most pixels are close to white (all channels > threshold)
    white_mask = np.all(tile > white_threshold, axis=-1)
    return white_mask.mean() > white_ratio


def load_valid_coordinates(output_dir: str) -> list:
    """
    Load valid coordinates from file if exists.

    Args:
        output_dir: Directory containing valid_coordinates.txt

    Returns:
        List of coordinate strings (e.g., ['0_0', '1024_0', ...]) or None if file not found
    """
    coord_file = os.path.join(output_dir, 'valid_coordinates.txt')
    if not os.path.exists(coord_file):
        return None

    coords = []
    with open(coord_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                coords.append(line)
    return coords


def save_valid_coordinates(output_dir: str, valid_coords: list, total_coords: int,
                          subject_name: str, white_threshold: int, white_ratio: float):
    """
    Save valid coordinates to a file.

    Args:
        output_dir: Directory to save the file
        valid_coords: List of valid coordinate strings
        total_coords: Total number of coordinates before filtering
        subject_name: Subject ID
        white_threshold: White threshold used for filtering
        white_ratio: White ratio used for filtering
    """
    coord_file = os.path.join(output_dir, 'valid_coordinates.txt')

    valid_count = len(valid_coords)
    percentage = (valid_count / total_coords * 100) if total_coords > 0 else 0

    with open(coord_file, 'w') as f:
        f.write(f"# Valid coordinates for {subject_name} (filtered: white_threshold={white_threshold}, white_ratio={white_ratio})\n")
        f.write(f"# Total: {valid_count} / {total_coords} tiles ({percentage:.1f}%)\n")
        for coord in valid_coords:
            f.write(f"{coord}\n")

    print(f"Saved valid coordinates to: {coord_file}")


def generate_tile_coordinates(height: int, width: int, patch_size: int = 1024,
                              stride: int = None) -> list:
    """
    Generate tile coordinates with configurable stride.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        patch_size: Size of each tile (default: 1024)
        stride: Step size between tiles (default: None = patch_size, i.e. non-overlapping)

    Returns:
        List of (x, y) coordinate tuples for tile top-left corners
    """
    if stride is None:
        stride = patch_size
    coords = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            coords.append((x, y))
    return coords


def load_gating_csv(gating_dir: str, subject_name: str) -> pd.DataFrame:
    """
    Load gating thresholds for a subject.

    Args:
        gating_dir: Directory containing gating CSV files
        subject_name: Subject ID (e.g., 'CRC01')

    Returns:
        DataFrame with channel names as index and columns: gate_start, gate_end, gate_active
    """
    csv_path = os.path.join(gating_dir, f'{subject_name}_gated_channel_ranges_sunni.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Gating CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.set_index('channel')
    return df


def extract_patch(channel_data: np.ndarray, x: int, y: int, patch_size: int) -> np.ndarray:
    """
    Extract a patch from channel data with boundary handling (zero-padding).
    Works for both 2D (H, W) and 3D (H, W, C) arrays.
    """
    if len(channel_data.shape) == 2:
        h, w = channel_data.shape
        is_rgb = False
    else:
        h, w, c = channel_data.shape
        is_rgb = True

    # Calculate valid region within image bounds
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(w, x + patch_size)
    y_end = min(h, y + patch_size)

    # Extract valid region
    valid_patch = channel_data[y_start:y_end, x_start:x_end]

    # Check if patch is complete (within bounds)
    if is_rgb:
        if valid_patch.shape[:2] == (patch_size, patch_size):
            return valid_patch
        padded = np.zeros((patch_size, patch_size, c), dtype=channel_data.dtype)
    else:
        if valid_patch.shape == (patch_size, patch_size):
            return valid_patch
        padded = np.zeros((patch_size, patch_size), dtype=channel_data.dtype)

    # Calculate where to place the valid region in the padded array
    pad_x_start = x_start - x
    pad_y_start = y_start - y

    if is_rgb:
        padded[pad_y_start:pad_y_start+valid_patch.shape[0],
               pad_x_start:pad_x_start+valid_patch.shape[1], :] = valid_patch
    else:
        padded[pad_y_start:pad_y_start+valid_patch.shape[0],
               pad_x_start:pad_x_start+valid_patch.shape[1]] = valid_patch

    return padded


def create_rgb_patch(
    patch_marker: np.ndarray,
    patch_epithelial: np.ndarray,
    patch_nuclear: np.ndarray
) -> np.ndarray:
    """
    Create RGB image from three channel patches.
    """
    h, w = patch_marker.shape
    patch_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    patch_rgb[:, :, 0] = patch_marker      # Red: Immune marker
    patch_rgb[:, :, 1] = patch_epithelial  # Green: Epithelial (Pan-CK)
    patch_rgb[:, :, 2] = patch_nuclear     # Blue: Nuclear (Hoechst)
    return patch_rgb


def find_multiplex_tiff(subject_dir: str) -> str:
    """
    Auto-detect the multiplex TIFF file in a subject directory.
    """
    # Priority 1: Large .ome.tiff files (real multiplex data)
    ome_tiff_files = glob.glob(os.path.join(subject_dir, '*.ome.tiff'))
    for f in ome_tiff_files:
        if 'mask' not in f.lower():
            return f

    # Priority 2: .ome.tif files that are NOT registered or mask
    ome_files = glob.glob(os.path.join(subject_dir, '*.ome.tif'))
    for f in ome_files:
        if 'mask' not in f.lower() and 'registered' not in f.lower():
            return f

    return None


def find_he_tiff(subject_dir: str) -> str:
    """
    Find the registered H&E TIFF file in a subject directory.

    The H&E file follows the pattern: *-registered.ome.tif

    Args:
        subject_dir: Path to subject directory (e.g., ./data/CRC01)

    Returns:
        Path to the registered H&E TIFF file, or None if not found
    """
    # Look for registered H&E file
    pattern = os.path.join(subject_dir, '*-registered.ome.tif')
    registered_files = glob.glob(pattern)

    if registered_files:
        # Return the first match (should typically be only one)
        return registered_files[0]

    return None


def process_tiles_hdf5(
    tif_path: str,
    output_dir: str,
    subject_name: str,
    gating_df: pd.DataFrame,
    target_columns: list = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = None,
    apply_clahe_flag: bool = False,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple = DEFAULT_CLAHE_TILE_GRID_SIZE,
    valid_coords: list = None
):
    """
    Process non-overlapping tiles from multiplex image and save to HDF5 format.
    Uses gating-based normalization (log transform).
    Label-free extraction - no label datasets created.

    Args:
        tif_path: Path to multiplex TIFF file
        output_dir: Output directory for HDF5 file
        subject_name: Subject ID
        gating_df: DataFrame with gating thresholds
        target_columns: List of markers to process
        patch_size: Tile size (default: 1024)
        apply_clahe_flag: Whether to apply CLAHE enhancement
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_grid_size: CLAHE tile grid size parameter
        valid_coords: List of valid coordinate strings to process (if None, process all)

    HDF5 Structure:
        tiles.h5
        ├── coordinates          # (N,) string - "x_y" format
        ├── shared_G             # (N, 1024, 1024) uint8 - Pan-CK
        ├── shared_B             # (N, 1024, 1024) uint8 - Hoechst
        ├── {marker}_R           # (N, 1024, 1024) uint8
        └── attributes
    """
    if target_columns is None:
        target_columns = TARGET_COL

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, 'tiles.h5')

    print(f"Loading multiplex TIFF: {tif_path}")
    tif = tifffile.TiffFile(tif_path)

    series_shape = tif.series[0].shape
    print(f"TIFF shape: {series_shape}")

    # Get image dimensions (assuming shape is (C, H, W) or similar)
    # The shape can be (C, H, W) or (H, W, C) depending on the TIFF
    # Typically for multiplex: (C, H, W)
    if len(series_shape) == 3:
        # Check which dimension is channels (smallest, likely < 50)
        if series_shape[0] < 50:
            # (C, H, W) format
            n_channels, image_height, image_width = series_shape
        else:
            # (H, W, C) format
            image_height, image_width, n_channels = series_shape
    else:
        # Handle unexpected shapes by reading first page
        page0 = tif.pages[0].asarray()
        image_height, image_width = page0.shape[:2]
        del page0

    print(f"Image dimensions: {image_height} x {image_width}")

    def load_channel(channel_idx):
        return tif.pages[channel_idx].asarray()

    # Generate tile coordinates
    if stride is None:
        stride = patch_size
    all_coords = generate_tile_coordinates(image_height, image_width, patch_size, stride)
    n_all_tiles = len(all_coords)

    n_rows = (image_height - patch_size) // stride + 1
    n_cols = (image_width - patch_size) // stride + 1

    overlap_desc = "non-overlapping" if stride == patch_size else f"overlap={patch_size - stride}"
    print(f"Generated {n_all_tiles} tiles ({n_rows} rows x {n_cols} cols)")
    print(f"Stride: {stride} ({overlap_desc})")

    # Filter to valid coordinates if provided
    if valid_coords is not None:
        valid_set = set(valid_coords)
        coords = [(x, y) for x, y in all_coords if f"{x}_{y}" in valid_set]
        print(f"Filtering to {len(coords)} valid coordinates (from {n_all_tiles} total)")
    else:
        coords = all_coords

    n_tiles = len(coords)

    # Convert coordinates to "x_y" format
    coord_strings = [f"{x}_{y}" for x, y in coords]

    if apply_clahe_flag:
        print(f"CLAHE enabled: clip_limit={clahe_clip_limit}, tile_grid_size={clahe_tile_grid_size}")

    # Load base channels
    print("Loading Hoechst channel...")
    nuclear_channel = load_channel(CH_MAP_RES_DICT['Hoechst'])
    print(f"  Hoechst shape: {nuclear_channel.shape}")

    print("Loading Pan-CK channel...")
    epithelial_channel = load_channel(CH_MAP_RES_DICT['Pan-CK'])

    # Get gating values for base channels
    print("Using gating thresholds...")
    nuclear_min = gating_df.loc['Hoechst', 'gate_start']
    nuclear_max = gating_df.loc['Hoechst', 'gate_end']
    epithelial_min = gating_df.loc['Pan-CK', 'gate_start']
    epithelial_max = gating_df.loc['Pan-CK', 'gate_end']

    print(f"  Hoechst: min={nuclear_min:.4f}, max={nuclear_max:.4f}")
    print(f"  Pan-CK:  min={epithelial_min:.4f}, max={epithelial_max:.4f}")

    # Create HDF5 file
    print(f"\nCreating HDF5 file: {h5_path}")

    with h5py.File(h5_path, 'w') as f:
        # Coordinates
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('coordinates', data=coord_strings, dtype=dt)

        # Shared channels
        chunk_size = min(100, n_tiles)
        f.create_dataset(
            'shared_G',
            shape=(n_tiles, patch_size, patch_size),
            dtype=np.uint8,
            chunks=(chunk_size, patch_size, patch_size)
        )
        f.create_dataset(
            'shared_B',
            shape=(n_tiles, patch_size, patch_size),
            dtype=np.uint8,
            chunks=(chunk_size, patch_size, patch_size)
        )

        # Per-marker datasets (R channel only, no labels)
        for col in target_columns:
            f.create_dataset(
                f'{col}_R',
                shape=(n_tiles, patch_size, patch_size),
                dtype=np.uint8,
                chunks=(chunk_size, patch_size, patch_size)
            )

        # First pass: extract shared channels (G, B)
        print("\nExtracting shared channels (G=Pan-CK, B=Hoechst)...")
        for idx, (x, y) in enumerate(tqdm(coords, desc="  Shared")):
            patch_nuclear = extract_patch(nuclear_channel, x, y, patch_size)
            patch_epithelial = extract_patch(epithelial_channel, x, y, patch_size)

            # Normalize using gating thresholds (log transform since gating values are in log scale)
            patch_nuclear_log = np.log(patch_nuclear.astype(float) + 1)
            patch_nuclear_norm = np.clip(
                (patch_nuclear_log - nuclear_min) / (nuclear_max - nuclear_min), 0, 1
            )
            patch_nuclear_norm = (255 * patch_nuclear_norm).astype(np.uint8)

            patch_epithelial_log = np.log(patch_epithelial.astype(float) + 1)
            patch_epithelial_norm = np.clip(
                (patch_epithelial_log - epithelial_min) / (epithelial_max - epithelial_min), 0, 1
            )
            patch_epithelial_norm = (255 * patch_epithelial_norm).astype(np.uint8)

            f['shared_G'][idx] = patch_epithelial_norm  # G = Pan-CK
            f['shared_B'][idx] = patch_nuclear_norm      # B = Hoechst

        # Second pass: extract each marker's R channel
        for col in target_columns:
            if col not in CH_MAP_RES_DICT:
                print(f"Warning: Channel '{col}' not found, skipping...")
                continue

            if col not in gating_df.index:
                print(f"Warning: Channel '{col}' not in gating CSV, skipping...")
                continue

            print(f"\nProcessing marker: {col}")
            marker_channel = load_channel(CH_MAP_RES_DICT[col])

            # Get gating values for marker
            marker_min = gating_df.loc[col, 'gate_start']
            marker_max = gating_df.loc[col, 'gate_end']
            print(f"  {col}: min={marker_min:.4f}, max={marker_max:.4f}")

            for idx, (x, y) in enumerate(tqdm(coords, desc=f"  {col}")):
                patch_marker = extract_patch(marker_channel, x, y, patch_size)

                # Normalize using gating thresholds (log transform since gating values are in log scale)
                patch_marker_log = np.log(patch_marker.astype(float) + 1)
                patch_marker_norm = np.clip(
                    (patch_marker_log - marker_min) / (marker_max - marker_min), 0, 1
                )
                patch_marker_norm = (255 * patch_marker_norm).astype(np.uint8)

                # Apply CLAHE if enabled (create RGB, apply CLAHE, extract R channel)
                if apply_clahe_flag:
                    # Create temporary RGB for CLAHE
                    temp_rgb = np.stack([patch_marker_norm,
                                        f['shared_G'][idx],
                                        f['shared_B'][idx]], axis=-1)
                    temp_rgb = apply_clahe(temp_rgb, clahe_clip_limit, clahe_tile_grid_size)
                    patch_marker_norm = temp_rgb[:, :, 0]

                f[f'{col}_R'][idx] = patch_marker_norm

            del marker_channel

        # Metadata
        f.attrs['subject'] = subject_name
        f.attrs['n_tiles'] = n_tiles
        f.attrs['patch_size'] = patch_size
        f.attrs['stride'] = stride
        f.attrs['image_height'] = image_height
        f.attrs['image_width'] = image_width
        f.attrs['biomarkers'] = target_columns
        f.attrs['normalization_method'] = 'gating'
        f.attrs['clahe_applied'] = apply_clahe_flag
        if apply_clahe_flag:
            f.attrs['clahe_clip_limit'] = clahe_clip_limit
            f.attrs['clahe_tile_grid_size'] = clahe_tile_grid_size

    tif.close()
    del nuclear_channel, epithelial_channel
    import gc
    gc.collect()

    # Report file size
    h5_size = os.path.getsize(h5_path) / (1024 ** 3)
    print(f"\nHDF5 file created: {h5_size:.2f} GB")
    print(f"Completed tile extraction for {subject_name}")


def process_tiles_png(
    tif_path: str,
    output_dir: str,
    subject_name: str,
    gating_df: pd.DataFrame,
    target_columns: list = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = None,
    skip_existing: bool = True,
    apply_clahe_flag: bool = False,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple = DEFAULT_CLAHE_TILE_GRID_SIZE,
    valid_coords: list = None
):
    """
    Process non-overlapping tiles from multiplex image and save as PNG files.
    Uses gating-based normalization (log transform).
    Label-free extraction - no pos/neg label in filename.

    Args:
        tif_path: Path to multiplex TIFF file
        output_dir: Output directory for PNG files
        subject_name: Subject ID
        gating_df: DataFrame with gating thresholds
        target_columns: List of markers to process
        patch_size: Tile size (default: 1024)
        skip_existing: Skip extraction if output file already exists
        apply_clahe_flag: Whether to apply CLAHE enhancement
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_grid_size: CLAHE tile grid size parameter
        valid_coords: List of valid coordinate strings to process (if None, process all)
    """
    if target_columns is None:
        target_columns = TARGET_COL

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading multiplex TIFF: {tif_path}")
    tif = tifffile.TiffFile(tif_path)

    series_shape = tif.series[0].shape
    print(f"TIFF shape: {series_shape}")

    # Get image dimensions
    if len(series_shape) == 3:
        if series_shape[0] < 50:
            n_channels, image_height, image_width = series_shape
        else:
            image_height, image_width, n_channels = series_shape
    else:
        page0 = tif.pages[0].asarray()
        image_height, image_width = page0.shape[:2]
        del page0

    print(f"Image dimensions: {image_height} x {image_width}")

    def load_channel(channel_idx):
        return tif.pages[channel_idx].asarray()

    # Generate tile coordinates
    if stride is None:
        stride = patch_size
    all_coords = generate_tile_coordinates(image_height, image_width, patch_size, stride)
    n_all_tiles = len(all_coords)

    n_rows = (image_height - patch_size) // stride + 1
    n_cols = (image_width - patch_size) // stride + 1

    overlap_desc = "non-overlapping" if stride == patch_size else f"overlap={patch_size - stride}"
    print(f"Generated {n_all_tiles} tiles ({n_rows} rows x {n_cols} cols)")
    print(f"Stride: {stride} ({overlap_desc})")

    # Filter to valid coordinates if provided
    if valid_coords is not None:
        valid_set = set(valid_coords)
        coords = [(x, y) for x, y in all_coords if f"{x}_{y}" in valid_set]
        print(f"Filtering to {len(coords)} valid coordinates (from {n_all_tiles} total)")
    else:
        coords = all_coords

    n_tiles = len(coords)

    if apply_clahe_flag:
        print(f"CLAHE enabled: clip_limit={clahe_clip_limit}, tile_grid_size={clahe_tile_grid_size}")

    # Load base channels
    print("Loading Hoechst channel...")
    nuclear_channel = load_channel(CH_MAP_RES_DICT['Hoechst'])
    print(f"  Hoechst shape: {nuclear_channel.shape}")

    print("Loading Pan-CK channel...")
    epithelial_channel = load_channel(CH_MAP_RES_DICT['Pan-CK'])

    # Get gating values for base channels
    print("Using gating thresholds...")
    nuclear_min = gating_df.loc['Hoechst', 'gate_start']
    nuclear_max = gating_df.loc['Hoechst', 'gate_end']
    epithelial_min = gating_df.loc['Pan-CK', 'gate_start']
    epithelial_max = gating_df.loc['Pan-CK', 'gate_end']

    print(f"  Hoechst: min={nuclear_min:.4f}, max={nuclear_max:.4f}")
    print(f"  Pan-CK:  min={epithelial_min:.4f}, max={epithelial_max:.4f}")

    # Process each marker
    for col in target_columns:
        if col not in CH_MAP_RES_DICT:
            print(f"Warning: Channel '{col}' not found, skipping...")
            continue

        if col not in gating_df.index:
            print(f"Warning: Channel '{col}' not in gating CSV, skipping...")
            continue

        print(f"\nProcessing marker: {col}")

        marker_output_dir = os.path.join(output_dir, col)
        os.makedirs(marker_output_dir, exist_ok=True)

        print(f"  Loading {col} channel...")
        marker_channel = load_channel(CH_MAP_RES_DICT[col])

        # Get gating values for marker
        marker_min = gating_df.loc[col, 'gate_start']
        marker_max = gating_df.loc[col, 'gate_end']
        print(f"  {col}: min={marker_min:.4f}, max={marker_max:.4f}")

        for x, y in tqdm(coords, desc=f"  {col}"):
            x_y = f"{x}_{y}"

            # No label in filename (label-free)
            img_path = os.path.join(
                marker_output_dir,
                f"{subject_name}_{col}_{x_y}.png"
            )

            if skip_existing and os.path.exists(img_path):
                continue

            patch_nuclear = extract_patch(nuclear_channel, x, y, patch_size)
            patch_epithelial = extract_patch(epithelial_channel, x, y, patch_size)
            patch_marker = extract_patch(marker_channel, x, y, patch_size)

            # Normalize using gating thresholds (log transform since gating values are in log scale)
            patch_nuclear_log = np.log(patch_nuclear.astype(float) + 1)
            patch_nuclear_norm = np.clip((patch_nuclear_log - nuclear_min) / (nuclear_max - nuclear_min), 0, 1)
            patch_nuclear_norm = (255 * patch_nuclear_norm).astype(np.uint8)

            patch_epithelial_log = np.log(patch_epithelial.astype(float) + 1)
            patch_epithelial_norm = np.clip((patch_epithelial_log - epithelial_min) / (epithelial_max - epithelial_min), 0, 1)
            patch_epithelial_norm = (255 * patch_epithelial_norm).astype(np.uint8)

            patch_marker_log = np.log(patch_marker.astype(float) + 1)
            patch_marker_norm = np.clip((patch_marker_log - marker_min) / (marker_max - marker_min), 0, 1)
            patch_marker_norm = (255 * patch_marker_norm).astype(np.uint8)

            patch_rgb = create_rgb_patch(patch_marker_norm, patch_epithelial_norm, patch_nuclear_norm)

            # Apply CLAHE if enabled
            if apply_clahe_flag:
                patch_rgb = apply_clahe(patch_rgb, clahe_clip_limit, clahe_tile_grid_size)

            Image.fromarray(patch_rgb).save(img_path)

        del marker_channel

    tif.close()
    del nuclear_channel, epithelial_channel
    import gc
    gc.collect()

    print(f"\nCompleted PNG tile extraction for {subject_name}")


def process_he_tiles(
    he_tif_path: str,
    output_dir: str,
    subject_name: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = None,
    skip_existing: bool = True,
    filter_empty: bool = False,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    white_ratio: float = DEFAULT_WHITE_RATIO
):
    """
    Extract H&E patches at non-overlapping grid coordinates.

    Extracts RGB patches from the registered H&E TIFF file at the same
    coordinates used for biomarker extraction, saving as PNG files.

    Args:
        he_tif_path: Path to the registered H&E TIFF file
        output_dir: Base output directory for the subject
        subject_name: Subject ID (e.g., 'CRC01')
        patch_size: Size of each tile (default: 1024)
        skip_existing: Skip extraction if output file already exists
        filter_empty: If True, skip empty (white background) tiles and save valid coordinates
        white_threshold: Pixel value threshold for empty detection (default: 240)
        white_ratio: Ratio of white pixels to consider tile as empty (default: 0.9)

    Returns:
        List of valid coordinate strings if filter_empty=True, else None
    """
    import gc

    he_output_dir = os.path.join(output_dir, 'HE')
    os.makedirs(he_output_dir, exist_ok=True)

    print(f"Loading H&E TIFF: {he_tif_path}")
    tif = tifffile.TiffFile(he_tif_path)

    # Load H&E image (RGB)
    he_image = tif.pages[0].asarray()
    print(f"H&E image shape: {he_image.shape}")

    # Get dimensions
    if len(he_image.shape) == 3:
        height, width, channels = he_image.shape
        print(f"H&E image: {height} x {width} x {channels} (RGB)")
    else:
        height, width = he_image.shape
        print(f"H&E image: {height} x {width} (grayscale)")

    # Generate tile coordinates
    if stride is None:
        stride = patch_size
    coords = generate_tile_coordinates(height, width, patch_size, stride)
    n_tiles = len(coords)

    n_rows = (height - patch_size) // stride + 1
    n_cols = (width - patch_size) // stride + 1

    overlap_desc = "non-overlapping" if stride == patch_size else f"overlap={patch_size - stride}"
    print(f"Generated {n_tiles} tiles ({n_rows} rows x {n_cols} cols)")
    print(f"Stride: {stride} ({overlap_desc})")
    print(f"Output directory: {he_output_dir}")

    if filter_empty:
        print(f"Empty tile filtering enabled: white_threshold={white_threshold}, white_ratio={white_ratio}")

    # Count existing files if skipping
    if skip_existing:
        existing_count = 0
        for x, y in coords:
            img_path = os.path.join(he_output_dir, f"{subject_name}_HE_{x}_{y}.png")
            if os.path.exists(img_path):
                existing_count += 1
        if existing_count > 0:
            print(f"Found {existing_count} existing files (will skip)")

    # Extract patches
    saved_count = 0
    skipped_count = 0
    empty_count = 0
    valid_coords_list = []

    for x, y in tqdm(coords, desc="H&E"):
        img_path = os.path.join(he_output_dir, f"{subject_name}_HE_{x}_{y}.png")

        if skip_existing and os.path.exists(img_path):
            skipped_count += 1
            # If file exists and filtering is enabled, we still count it as valid
            if filter_empty:
                valid_coords_list.append(f"{x}_{y}")
            continue

        patch = extract_patch(he_image, x, y, patch_size)

        # Check if tile is empty (white background)
        if filter_empty and len(patch.shape) == 3:
            if is_tile_empty(patch, white_threshold, white_ratio):
                empty_count += 1
                continue

        Image.fromarray(patch).save(img_path)
        saved_count += 1
        if filter_empty:
            valid_coords_list.append(f"{x}_{y}")

    tif.close()
    del he_image
    gc.collect()

    print(f"\nH&E extraction complete for {subject_name}")
    print(f"  Saved: {saved_count} tiles")
    print(f"  Skipped (existing): {skipped_count} tiles")
    if filter_empty:
        print(f"  Filtered (empty): {empty_count} tiles")
        print(f"  Valid tiles: {len(valid_coords_list)} / {n_tiles} ({len(valid_coords_list)/n_tiles*100:.1f}%)")
        # Save valid coordinates to file
        save_valid_coordinates(output_dir, valid_coords_list, n_tiles,
                              subject_name, white_threshold, white_ratio)
        return valid_coords_list
    else:
        print(f"  Total: {n_tiles} tiles")
        return None


def process_subject(
    subject_name: str,
    tiff_base_dir: str,
    output_base_dir: str,
    gating_dir: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    stride: int = None,
    skip_existing: bool = True,
    output_format: str = 'hdf5',
    target_columns: list = None,
    mode: str = 'biomarker',
    apply_clahe: bool = False,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size: tuple = DEFAULT_CLAHE_TILE_GRID_SIZE,
    filter_empty: bool = False,
    white_threshold: int = DEFAULT_WHITE_THRESHOLD,
    white_ratio: float = DEFAULT_WHITE_RATIO
):
    """
    Process a single subject - extract non-overlapping tiles.

    Args:
        subject_name: Subject ID (e.g., 'CRC01')
        tiff_base_dir: Base directory for TIFF files
        output_base_dir: Base output directory for tiles
        gating_dir: Directory containing gating CSV files
        patch_size: Tile size (default: 1024)
        skip_existing: Skip existing files (PNG mode only)
        output_format: 'hdf5' or 'png'
        target_columns: Specific markers to process
        mode: 'biomarker', 'he', or 'both'
        apply_clahe: Whether to apply CLAHE enhancement to biomarker images
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_grid_size: CLAHE tile grid size parameter
        filter_empty: Whether to filter empty tiles during H&E extraction
        white_threshold: Pixel value threshold for empty detection
        white_ratio: Ratio of white pixels to consider tile as empty
    """
    subject_dir = os.path.join(tiff_base_dir, subject_name)
    output_dir = os.path.join(output_base_dir, subject_name)

    print(f"\n{'='*60}")
    print(f"Processing: {subject_name} (mode: {mode}, format: {output_format})")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Check subject directory exists
    if not os.path.exists(subject_dir):
        print(f"Warning: Subject directory not found: {subject_dir}")
        return

    valid_coords = None

    # Process H&E if requested (do this first to get valid coordinates)
    if mode in ('he', 'both'):
        he_tif = find_he_tiff(subject_dir)
        if he_tif:
            print(f"H&E TIFF: {he_tif}")
            valid_coords = process_he_tiles(
                he_tif_path=he_tif,
                output_dir=output_dir,
                subject_name=subject_name,
                patch_size=patch_size,
                stride=stride,
                skip_existing=skip_existing,
                filter_empty=filter_empty,
                white_threshold=white_threshold,
                white_ratio=white_ratio
            )
        else:
            print(f"Warning: No registered H&E TIFF found for {subject_name}")

    # Process biomarkers if requested
    if mode in ('biomarker', 'both'):
        # Load gating CSV
        try:
            gating_df = load_gating_csv(gating_dir, subject_name)
            print(f"Loaded gating CSV for {subject_name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        # Find multiplex TIFF
        multiplex_tif = find_multiplex_tiff(subject_dir)
        if not multiplex_tif:
            print(f"Warning: No multiplex TIFF found for {subject_name}")
            return

        print(f"Multiplex TIFF: {multiplex_tif}")

        # Try to load valid coordinates if not already available (from previous H&E run)
        if valid_coords is None:
            valid_coords = load_valid_coordinates(output_dir)
            if valid_coords is not None:
                print(f"Loaded {len(valid_coords)} valid coordinates from file")

        # Process based on output format
        if output_format == 'hdf5':
            process_tiles_hdf5(
                tif_path=multiplex_tif,
                output_dir=output_dir,
                subject_name=subject_name,
                gating_df=gating_df,
                target_columns=target_columns,
                patch_size=patch_size,
                stride=stride,
                apply_clahe_flag=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                valid_coords=valid_coords
            )
        else:
            process_tiles_png(
                tif_path=multiplex_tif,
                output_dir=output_dir,
                subject_name=subject_name,
                gating_df=gating_df,
                target_columns=target_columns,
                patch_size=patch_size,
                stride=stride,
                skip_existing=skip_existing,
                apply_clahe_flag=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid_size=clahe_tile_grid_size,
                valid_coords=valid_coords
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract non-overlapping tiles from multiplex images (label-free, gating-based normalization).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all markers for CRC01 in HDF5 format
  python patch_extraction_nolabel.py --subjects CRC01 --output-format hdf5

  # Extract specific markers
  python patch_extraction_nolabel.py --subjects CRC01 --markers PD-L1 CD45

  # Multiple subjects
  python patch_extraction_nolabel.py --subjects CRC01 CRC02 CRC03

  # PNG output
  python patch_extraction_nolabel.py --subjects CRC01 --output-format png

  # Extract H&E patches only
  python patch_extraction_nolabel.py --subjects CRC01 --mode he

  # Extract both biomarkers and H&E
  python patch_extraction_nolabel.py --subjects CRC01 --mode both

  # Recommended workflow: First H&E with filtering, then biomarker with CLAHE
  python patch_extraction_nolabel.py --subjects CRC01 --mode he --filter-empty
  python patch_extraction_nolabel.py --subjects CRC01 --mode biomarker --clahe

  # One-step: H&E + biomarker with filtering and CLAHE
  python patch_extraction_nolabel.py --subjects CRC01 --mode both --filter-empty --clahe

  # Custom CLAHE parameters
  python patch_extraction_nolabel.py --subjects CRC01 --clahe --clahe-clip-limit 3.0

  # Custom empty filtering parameters (more lenient)
  python patch_extraction_nolabel.py --subjects CRC01 --mode he --filter-empty --white-ratio 0.95

Key differences from patch_extraction_gating.py:
  - No label CSV required (generates coordinates from image dimensions)
  - Non-overlapping tiles (stride = patch_size)
  - No label datasets in HDF5 output
  - Fewer tiles: ~4,180 vs ~16,642 for CRC01

New features:
  - CLAHE enhancement for biomarker images (--clahe)
  - Empty tile filtering for H&E (--filter-empty)
  - Valid coordinates are saved and reused for biomarker extraction
        """
    )

    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=None,
        help='Subject IDs to process (default: all available)'
    )

    parser.add_argument(
        '--markers',
        type=str,
        nargs='+',
        default=None,
        help='Specific markers to process (default: all 13 immune markers)'
    )

    parser.add_argument(
        '--tiff-dir',
        type=str,
        default='./data',
        help='Base directory for TIFF files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Base output directory for tiles'
    )

    parser.add_argument(
        '--gating-dir',
        type=str,
        default=DEFAULT_GATING_DIR,
        help='Directory containing gating CSV files'
    )

    parser.add_argument(
        '--patch-size',
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help='Tile size (default: 1024)'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=None,
        help='Stride between tiles (default: patch_size, i.e. non-overlapping). '
             'Set smaller than patch_size for overlap (e.g. --patch-size 1024 --stride 512 = 50%% overlap)'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['hdf5', 'png'],
        default='hdf5',
        help='Output format: hdf5 (default) or png'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip existing files (re-extract all, PNG mode only)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['biomarker', 'he', 'both'],
        default='biomarker',
        help='Extraction mode: biomarker (default), he (H&E only), or both'
    )

    # CLAHE arguments
    parser.add_argument(
        '--clahe',
        action='store_true',
        help='Enable CLAHE enhancement for biomarker images'
    )

    parser.add_argument(
        '--clahe-clip-limit',
        type=float,
        default=DEFAULT_CLAHE_CLIP_LIMIT,
        help=f'CLAHE clip limit (default: {DEFAULT_CLAHE_CLIP_LIMIT})'
    )

    parser.add_argument(
        '--clahe-tile-grid',
        type=int,
        default=DEFAULT_CLAHE_TILE_GRID_SIZE[0],
        help=f'CLAHE tile grid size (default: {DEFAULT_CLAHE_TILE_GRID_SIZE[0]})'
    )

    # Empty tile filtering arguments
    parser.add_argument(
        '--filter-empty',
        action='store_true',
        help='Enable empty tile filtering for H&E extraction'
    )

    parser.add_argument(
        '--white-threshold',
        type=int,
        default=DEFAULT_WHITE_THRESHOLD,
        help=f'White pixel threshold for empty detection (default: {DEFAULT_WHITE_THRESHOLD})'
    )

    parser.add_argument(
        '--white-ratio',
        type=float,
        default=DEFAULT_WHITE_RATIO,
        help=f'White pixel ratio to consider tile as empty (default: {DEFAULT_WHITE_RATIO})'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-scan subjects from tiff-dir if not specified
    if args.subjects:
        subjects = args.subjects
    else:
        if os.path.isdir(args.tiff_dir):
            subjects = sorted([
                d for d in os.listdir(args.tiff_dir)
                if os.path.isdir(os.path.join(args.tiff_dir, d))
            ])
        else:
            subjects = []
        if not subjects:
            print(f"Error: No subject directories found in {args.tiff_dir}")
            return

    target_markers = args.markers if args.markers else TARGET_COL

    # Resolve stride
    stride = args.stride if args.stride is not None else args.patch_size

    # Convert tile grid size to tuple
    clahe_tile_grid_size = (args.clahe_tile_grid, args.clahe_tile_grid)

    print("="*60)
    print("Label-Free Tile Extraction")
    print("="*60)
    print(f"Subjects: {subjects}")
    print(f"Mode: {args.mode}")
    if args.mode in ('biomarker', 'both'):
        print(f"Markers: {target_markers}")
        print(f"CLAHE: {args.clahe}")
        if args.clahe:
            print(f"  Clip limit: {args.clahe_clip_limit}")
            print(f"  Tile grid: {clahe_tile_grid_size}")
    if args.mode in ('he', 'both'):
        print(f"Filter empty: {args.filter_empty}")
        if args.filter_empty:
            print(f"  White threshold: {args.white_threshold}")
            print(f"  White ratio: {args.white_ratio}")
    print(f"Patch size: {args.patch_size}")
    overlap_desc = "non-overlapping" if stride == args.patch_size else f"overlap={args.patch_size - stride}"
    print(f"Stride: {stride} ({overlap_desc})")
    print(f"Output format: {args.output_format}")
    print(f"Skip existing: {not args.no_skip}")
    print(f"Gating directory: {args.gating_dir}")
    print(f"Output directory: {args.output_dir}")

    for subject_name in subjects:
        process_subject(
            subject_name=subject_name,
            tiff_base_dir=args.tiff_dir,
            output_base_dir=args.output_dir,
            gating_dir=args.gating_dir,
            patch_size=args.patch_size,
            stride=stride,
            skip_existing=not args.no_skip,
            output_format=args.output_format,
            target_columns=target_markers,
            mode=args.mode,
            apply_clahe=args.clahe,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            filter_empty=args.filter_empty,
            white_threshold=args.white_threshold,
            white_ratio=args.white_ratio
        )

    print("\n" + "="*60)
    print("All processing completed!")
    print("="*60)


if __name__ == '__main__':
    main()
