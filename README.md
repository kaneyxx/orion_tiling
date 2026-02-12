# Orion Tiling

Label-free tile extraction from multiplex TIFF images with gating-based normalization.

Extracts RGB tiles where:
- **R** = immune marker (one per biomarker)
- **G** = Pan-CK (epithelial, shared)
- **B** = Hoechst (nuclear, shared)

## Quick Start

```bash
pip install -r requirements.txt

# Extract all subjects in tiff-dir as HDF5 (default)
python patch_extraction_nolabel.py

# Extract specific subjects
python patch_extraction_nolabel.py --subjects CRC01 CRC02

# Extract H&E + biomarkers with filtering and CLAHE
python patch_extraction_nolabel.py --subjects CRC01 --mode both --filter-empty --clahe
```

## Settings You Need to Adjust

### A. Path Settings (CLI arguments — most important)

| Argument | Description | Default |
|----------|-------------|---------|
| `--tiff-dir` | Directory containing subject subdirectories with multiplex TIFFs | `./data` |
| `--output-dir` | Output directory for extracted tiles | `./output` |
| `--gating-dir` | Directory containing gating CSV files | `./gating` |

Expected directory structure:
```
./data/
├── CRC01/
│   ├── *.ome.tiff           # multiplex TIFF
│   └── *-registered.ome.tif # registered H&E (optional)
├── CRC02/
│   └── ...

./gating/
├── CRC01_gated_channel_ranges_sunni.csv
├── CRC02_gated_channel_ranges_sunni.csv
└── ...
```

### B. Tile Size & Overlap (CLI arguments)

| Argument | Description | Default |
|----------|-------------|---------|
| `--patch-size` | Tile size in pixels | `1024` |
| `--stride` | Step size between tiles | `patch_size` (non-overlapping) |

Set `--stride` smaller than `--patch-size` for overlapping tiles:
```bash
# 50% overlap
python patch_extraction_nolabel.py --patch-size 1024 --stride 512

# Non-overlapping (default)
python patch_extraction_nolabel.py --patch-size 1024
```

### C. Channel / Biomarker Settings (modify in code)

These lists are defined at the top of `patch_extraction_nolabel.py` and must match your multiplex panel:

- **`CH_LI`** (line ~60): All channel names in your multiplex TIFF, in order.
  ```python
  CH_LI = [
      'Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4',
      'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163',
      'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA'
  ]
  ```

- **`TARGET_COL`** (line ~71): Biomarkers to extract (subset of `CH_LI`).
  ```python
  TARGET_COL = [
      'CD45', 'CD31', 'CD68', 'CD4', 'FOXP3', 'CD8a',
      'CD45RO', 'CD20', 'PD-L1', 'CD3e', 'CD163', 'PD-1', 'Ki67'
  ]
  ```

You can also override `TARGET_COL` at runtime:
```bash
python patch_extraction_nolabel.py --markers PD-L1 CD45 CD68
```

### D. CLAHE Enhancement (CLI arguments, optional)

| Argument | Description | Default |
|----------|-------------|---------|
| `--clahe` | Enable CLAHE enhancement | disabled |
| `--clahe-clip-limit` | Contrast limit | `2.0` |
| `--clahe-tile-grid` | Grid size for local histograms | `8` |

### E. Empty Tile Filtering (CLI arguments, optional)

| Argument | Description | Default |
|----------|-------------|---------|
| `--filter-empty` | Enable empty tile filtering (H&E mode) | disabled |
| `--white-threshold` | Pixel value threshold for "white" | `240` |
| `--white-ratio` | Ratio of white pixels to skip tile | `0.9` |

Filtered coordinates are saved to `valid_coordinates.txt` and reused for biomarker extraction.

### F. Output Format & Mode (CLI arguments)

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-format` | `hdf5` or `png` | `hdf5` |
| `--mode` | `biomarker`, `he`, or `both` | `biomarker` |
| `--no-skip` | Re-extract existing files (PNG mode) | skip existing |

## Usage Examples

```bash
# Recommended workflow: filter empty tiles first, then extract biomarkers
python patch_extraction_nolabel.py --subjects CRC01 --mode he --filter-empty
python patch_extraction_nolabel.py --subjects CRC01 --mode biomarker --clahe

# One-step: both modes with all options
python patch_extraction_nolabel.py --subjects CRC01 --mode both --filter-empty --clahe

# All subjects (auto-detected from tiff-dir)
python patch_extraction_nolabel.py --mode both --filter-empty

# PNG output instead of HDF5
python patch_extraction_nolabel.py --subjects CRC01 --output-format png

# Custom paths
python patch_extraction_nolabel.py \
    --tiff-dir /path/to/tiffs \
    --output-dir /path/to/output \
    --gating-dir /path/to/gating \
    --subjects CRC01
```

## HDF5 Output Structure

```
{subject}/tiles.h5
├── coordinates          # (N,) string — "x_y" format
├── shared_G             # (N, H, W) uint8 — Pan-CK
├── shared_B             # (N, H, W) uint8 — Hoechst
├── {marker}_R           # (N, H, W) uint8 — per biomarker
└── attributes:
    ├── subject
    ├── n_tiles
    ├── patch_size
    ├── stride
    ├── image_height, image_width
    ├── biomarkers
    ├── normalization_method: 'gating'
    └── clahe_applied
```

## Normalization Method

Gating-based log-transform normalization:

1. Apply `log(pixel + 1)` to raw intensity
2. Clip to `[gate_start, gate_end]` from per-subject gating CSV
3. Rescale to `[0, 255]` uint8

This normalizes each channel's dynamic range per subject using expert-defined gating thresholds.
