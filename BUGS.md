# Code Review: Bugs & Issues

## 1. Depth relation losses are swapped — `training/graph_modules.py:266-269`

**Severity: Critical**

The Depth Anything V2 model outputs inverse depth (documented in `scop_depth/depth.py:278`: *"larger values behave like stronger inverse depth, so we interpret larger values as 'closer'"*). After the transformation in `graph_targets.py`, `z = depth_value * 2 - 1`, so **+1 = closest, -1 = farthest**.

With that convention the depth relation losses are backwards:

| Relation | Meaning | Expected `delta[2]` | Current loss | Correct loss |
|---|---|---|---|---|
| `in_front_of` | A closer, A.z > B.z | delta < 0 | `relu(0.05 - delta)` ❌ always fires | `relu(0.05 + delta)` |
| `behind` | A farther, A.z < B.z | delta > 0 | `relu(0.05 + delta)` ❌ always fires | `relu(0.05 - delta)` |
| `hidden_by` | B closer (hides A), delta > 0 | delta > 0 | `relu(0.05 - delta)` ✓ correct | — |

`in_front_of` and `behind` are effectively swapped. The model is penalized for having the *correct* depth ordering on every training step.

---

## 2. Hardcoded two-object depth limit — `training/graph_targets.py:21-24`

**Severity: High**

```python
depth_values = [
    float(depth["bbox1"]["median"]) if depth else 0.0,
    float(depth["bbox2"]["median"]) if depth else 0.0,
]
```

Only `bbox1` and `bbox2` are read. If a scene has ≥3 annotations, every object at index 2+ silently gets `cz = 0.0` (line 31). There is no warning, and those slots still contribute to the geometry loss with a wrong target.

---

## 3. Reversed message passing — `training/graph_modules.py:52-59`

**Severity: Medium**

```python
messages = self.message_mlp(
    torch.cat([node_states[dst], relation_embeddings], dim=-1)
)
aggregated.index_add_(0, src, messages)
```

For edge `(src → dst, relation)`, the message is computed from **`dst`'s** state and added to **`src`**. Standard GNN convention goes the other direction (neighbor sends to current node). Since bidirectional inverse edges are added the information still flows both ways, but the relation embeddings' semantics are inverted — the `"left_of"` embedding is used to update A using B's state, rather than the `"right_of"` embedding that B's perspective on A would call for. This contradicts the intent of `_initialize_relation_embeddings`.

---

## 4. `"on"` relation missing directionality — `training/graph_modules.py:270-271`

**Severity: Medium**

```python
elif relation == "on":
    sample_losses.append(F.relu(delta[1].abs() - 0.2))
```

"A on B" means A rests on top of B — A should appear **above** B in the image (`delta[1] > 0` since y increases downward in image coordinates). The current loss only enforces vertical *proximity*, not direction. Compare `"above"` which correctly uses `relu(0.1 - delta[1])` to enforce that A is higher than B. The `"on"` loss should do the same.

---

## 5. Missing `INVERSE_RELATION` coverage check — `training/scene_graph.py:66`

**Severity: Low**

`INVERSE_RELATION[relation]` has no fallback. If a new relation is added to `RELATION_VOCAB` but omitted from `INVERSE_RELATION`, the error surfaces as a `KeyError` deep inside the DataLoader with no useful message. A guard at import time (e.g., `assert set(INVERSE_RELATION) == set(RELATION_VOCAB)`) would catch this immediately.
