# relation-aware-slot-attention

This workspace currently contains `SCOP-Depth`, a depth-augmented variant of the SCOP data pipeline used to extract 2D, depth-order, and occlusion-style relations from COCO.

The implementation lives in [scop_depth](/Users/newbieted/workspace/relation-aware-slot-attention/scop_depth), and the module path is `scop_depth`.

For evaluation, this repo now keeps only our wrapper scripts and a lightweight
override layer. External benchmark code such as T2I-CompBench should be prepared
outside the tracked source tree with:

```bash
./scripts/setup/bootstrap_all.sh
```

This creates a local `.venv`, installs the repo and evaluation dependencies,
prepares the external T2I-CompBench checkout, and downloads the benchmark
weights.

If you previously used an older bootstrap version that created
`scripts/.venv`, remove that stale environment first.
