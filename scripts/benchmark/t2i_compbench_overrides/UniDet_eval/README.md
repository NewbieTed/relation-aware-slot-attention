# UniDet Eval Overrides

This directory mirrors the `UniDet_eval` portion of T2I-CompBench for local
patching. It contains the 2D and 3D spatial scoring entrypoints plus the helper
modules they import.

## Files

- `2D_spatial_eval.py`: patched 2D spatial evaluator.
- `3D_spatial_eval.py`: patched 3D spatial evaluator.
- `experts/`: detector and depth helper overrides used by those evaluators.

## Usage

These files are copied into `external/T2I-CompBench/UniDet_eval` by the benchmark
wrapper. Run the wrapper from the repo root rather than invoking these files
directly.
