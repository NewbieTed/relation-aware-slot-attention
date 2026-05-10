# Evaluation And Debugging

This FLUX branch keeps evaluation code that is still useful for the current
architecture:

- `evaluation.generate_flux_relation`: generate one FLUX image from a prompt,
  using the frozen GNN to render an OSCR condition image.
- `evaluation.generate_flux_relation_t2i`: generate T2I-CompBench prompt-file
  samples with GNN-predicted OSCR conditions and the released SeeThrough3D LoRA.
- `evaluation.debug_gnn_prompt`: print detailed GNN/CVAE message-passing and
  layout-head traces for a prompt.
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
python3 -m evaluation.generate_flux_relation_t2i \
  --config configs/flux/eval_official_seethrough3d_spatial_20p_bf16_512.yaml
```

Example GNN/CVAE trace:

```bash
python3 -m evaluation.debug_gnn_prompt \
  --prompt "a suitcase in front of an apple" \
  --graph-encoder-path outputs/train/graph_pretrain_flux_3dbox_cvae/final/graph_encoder.pt \
  --output-dir outputs/debug/gnn_cvae_trace \
  --layout-sample-mode prior_sample \
  --seed 42
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
