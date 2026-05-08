# UniDet Expert Overrides

This directory contains helper modules used by the patched T2I-CompBench
spatial evaluators.

## Subdirectories

- `depth/`: depth dataset generation helper used by 3D spatial scoring.
- `obj_detection/`: object-detection dataset helpers used by 2D and 3D scoring.

`model_bank_3d.py` wires the local detector/depth models expected by the 3D
spatial evaluator.
