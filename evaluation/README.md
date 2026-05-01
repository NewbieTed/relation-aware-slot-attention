# Evaluation And Debugging

This FLUX branch keeps evaluation code that is still useful for the current
architecture:

- `evaluation.generate_flux_relation`: generate one FLUX image from a prompt,
  using the frozen GNN to render an OSCR condition image.
- `evaluation.debug_gnn_prompt`: print detailed GNN message-passing traces for a prompt.
- `evaluation.visualize_gnn_layout`: visualize predicted GNN centers and box regions.
- `evaluation.t2i_compbench`: score an existing generated samples directory with
  a prepared T2I-CompBench checkout.
- `evaluation.geneval` and `evaluation.visor`: wrappers for existing generated
  outputs when those benchmarks are needed.

The old Stable Diffusion generator and slot-attention inspection tools have been
removed from this branch. If we need those experiments again, switch back to the
SD branch rather than reintroducing them here.

Example FLUX generation:

```bash
python3 -m evaluation.generate_flux_relation \
  --prompt "a cat to the left of a dog" \
  --checkpoint-dir outputs/train/flux1dev_oscr_lora128/final \
  --output-dir outputs/debug/flux_cat_left_dog
```

Example scoring of an already generated T2I-CompBench directory:

```bash
python3 -m evaluation.t2i_compbench \
  --benchmark spatial \
  --t2i-compbench-root external/T2I-CompBench \
  --generated-dir outputs/eval/my_flux_spatial_run \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --python-bin python3
```
