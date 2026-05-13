# Object Detection Expert Overrides

This directory contains patched T2I-CompBench object-detection helpers.

## Files

- `generate_dataset.py`: prepares generated samples for 2D spatial detection.
- `generate_dataset_3d.py`: prepares generated samples for 3D spatial detection.
- `utils.py`: shared helper functions for the patched detection scripts.

These files exist so the external benchmark can score our generated outputs
consistently. They are not used during GNN training or FLUX generation.
