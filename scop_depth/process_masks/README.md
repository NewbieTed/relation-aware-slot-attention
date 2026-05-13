# Process Masks

This subpackage adds optional object masks to a generated SCOP-Depth dataset.
It is mostly useful for older Stable Diffusion experiments and mask inspection;
the current FLUX/SeeThrough3D path does not require SAM masks.

## Usage

```bash
python3 -m scop_depth.process_masks /path/to/scop_depth_dataset
```

The command expects a SCOP-Depth output directory containing `metadata.jsonl`
and exported images. It writes mask artifacts back into that dataset directory.
