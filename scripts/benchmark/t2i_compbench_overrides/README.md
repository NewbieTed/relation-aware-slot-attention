# T2I-CompBench Overrides

This directory contains local replacement files for T2I-CompBench. The wrappers
copy these files into the external benchmark checkout before scoring so that the
benchmark works with the current environment and our spatial/3D evaluation
needs.

These files are compatibility patches, not core model code. Keep edits minimal
and document why an override is needed.

## Usage

The override copy happens inside:

```bash
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

You normally should not call files in this directory directly.
