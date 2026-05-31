# Triple-CVAE GNN Layout Architecture

This note explains the current graph layout model as if a peer is joining the
project and wants to understand exactly how a prompt becomes predicted 3D object
boxes. The goal of this module is to predict object layouts that can later be
rendered into OSCR images for SeeThrough3D/FLUX inference.

The model is inspired by 3D_SLN's scene-graph-to-scene VAE. The active
3D_SLN-aligned CVAE configs use both a scene-level latent and per-object latent
variables, while deterministic GNN baselines use the same scene graph input but
skip latent sampling entirely.

As of the latest experiments, this architecture is evaluated directly as a
layout predictor. The current paper-facing table is `docs/paper_table.md`.
Image generation through SeeThrough3D/FLUX remains useful qualitatively, but the
main quantitative comparison is now relation accuracy, reference-box quality,
layout validity, and stochastic diversity on held-out SCOP-Depth metadata.

## Inputs

Each training example contains a prompt and processed object annotations.

For example:

```text
a suitcase in front of an apple
```

From this example, preprocessing gives us:

```text
object labels: suitcase, apple
relation: suitcase in_front_of apple
ground-truth boxes: one normalized 3D box per object
```

The ground-truth box for object `i` is:

```text
gt_box_i = [x0_i, y0_i, z0_i, x1_i, y1_i, z1_i]
```

`gt_box_i` means the target 3D bounding box for object `i`.

`x0_i, y0_i, z0_i` are the minimum normalized coordinates of object `i`.

`x1_i, y1_i, z1_i` are the maximum normalized coordinates of object `i`.

All six values are normalized into `[0, 1]`.

For the suitcase example:

```text
gt_box_suitcase = [x0, y0, z0, x1, y1, z1]
gt_box_apple    = [x0, y0, z0, x1, y1, z1]
```

The exact values depend on the cropped training image and depth estimate.

## Scene Graph

The prompt is represented as a scene graph.

A scene graph has nodes and edges:

```text
node_i = one object in the prompt
edge_ij = one relation from object i to object j
```

For:

```text
a suitcase in front of an apple
```

we build:

```text
nodes:
  node_0 = suitcase
  node_1 = apple

edges:
  edge_01 = suitcase in_front_of apple
  edge_10 = apple behind suitcase
```

`edge_10` is added because we use bidirectional message passing. If the prompt
says "suitcase in front of apple", then the inverse edge says "apple behind
suitcase".

## Object Label Embeddings

Each object label is embedded before graph message passing.

```text
label_embedding_i = text_encoder(object_label_i)
```

`label_embedding_i` means the text embedding for object `i`'s name.

For example:

```text
label_embedding_suitcase = text_encoder("suitcase")
label_embedding_apple    = text_encoder("apple")
```

These label embeddings are projected into the GNN hidden dimension:

```text
original_node_i = node_proj(label_embedding_i)
```

`original_node_i` means the initial hidden vector for object `i` before the
triple-GNN updates. It contains semantic information from the object label, such
as "suitcase" or "apple", but it does not yet contain graph-contextual layout
information.

## Relation Embeddings

Each relation label is also embedded.

```text
relation_embedding_ij = relation_embedding_table[relation_ij]
```

`relation_ij` means the relation from object `i` to object `j`.

For example:

```text
relation_01 = in_front_of
relation_10 = behind
```

The relation embedding is projected into the same hidden dimension used by edge
states:

```text
edge_state_ij = edge_proj(relation_embedding_ij)
```

`edge_state_ij` means the hidden vector for the edge from object `i` to object
`j`.

## Prior Triple-GNN

The prior graph pass uses only the object labels and relation labels. It does
not see the ground-truth boxes.

The initial prior node state is:

```text
prior_node_i^0 = original_node_i
```

`prior_node_i^0` means the starting prior node state for object `i` at GNN layer
0.

The initial prior edge state is:

```text
prior_edge_ij^0 = edge_state_ij
```

`prior_edge_ij^0` means the starting prior edge state for the relation from
object `i` to object `j`.

For each triple-GNN layer, the model processes every triple:

```text
triple_ij = concat(prior_node_i, prior_edge_ij, prior_node_j)
```

`triple_ij` means the combined vector containing the subject object state, the
relation edge state, and the target object state.

For the suitcase example:

```text
triple_01 = concat(prior_node_suitcase, prior_edge_in_front_of, prior_node_apple)
triple_10 = concat(prior_node_apple, prior_edge_behind, prior_node_suitcase)
```

Each triple is passed through an MLP:

```text
triple_output_ij = triple_mlp(triple_ij)
```

`triple_output_ij` means the updated information computed from the full
subject-relation-object triple.

The output is split into three parts:

```text
subject_msg_i, edge_update_ij, object_msg_j = split(triple_output_ij)
```

`subject_msg_i` means the message sent back to the subject object.

`edge_update_ij` means the update to the relation edge state.

`object_msg_j` means the message sent to the target object.

Each object may appear in multiple triples, so messages for each object are
aggregated:

```text
aggregated_msg_i = mean(all messages sent to object i)
```

`aggregated_msg_i` means the average message collected by object `i` from all
triples where it appears as either subject or object.

The node state is updated with a residual connection:

```text
prior_node_i_next = prior_node_i + node_update(concat(prior_node_i, aggregated_msg_i))
```

`prior_node_i_next` means the updated prior node state for object `i` after this
triple-GNN layer.

The edge state is also updated with a residual connection:

```text
prior_edge_ij_next = prior_edge_ij + edge_update_ij
```

`prior_edge_ij_next` means the updated edge state for relation `i -> j` after
this layer.

This process repeats for `gnn_layers` layers.

After the prior triple-GNN, we have:

```text
prior_node_i = final prior node state for object i
prior_edge_ij = final prior edge state for relation i -> j
```

These states represent what the model can infer from the scene graph alone.

## Posterior Triple-GNN

The posterior graph pass sees the ground-truth boxes during training. This is
what makes the model a conditional VAE.

First, each ground-truth box is embedded:

```text
layout_feature_i = gt_layout_encoder(gt_box_i)
```

`layout_feature_i` means the learned feature vector produced from object `i`'s
ground-truth 3D box.

The posterior node input is:

```text
posterior_node_i^0 = posterior_node_init(concat(original_node_i, layout_feature_i))
```

`posterior_node_i^0` means the starting posterior node state for object `i`.
Unlike `prior_node_i^0`, this state has access to the ground-truth box.

The posterior edge state starts from the same relation edge state:

```text
posterior_edge_ij^0 = edge_state_ij
```

Then we run the same triple-GNN update pattern:

```text
posterior_triple_ij = concat(posterior_node_i, posterior_edge_ij, posterior_node_j)
posterior_triple_output_ij = triple_mlp(posterior_triple_ij)
subject_msg_i, edge_update_ij, object_msg_j = split(posterior_triple_output_ij)
posterior_node_i_next = posterior_node_i + node_update(concat(posterior_node_i, aggregated_msg_i))
posterior_edge_ij_next = posterior_edge_ij + edge_update_ij
```

After the posterior triple-GNN, we have:

```text
posterior_node_i = final posterior node state for object i
posterior_edge_ij = final posterior edge state for relation i -> j
```

These states represent what the model can infer when it sees both the scene
graph and the true training layout.

## Scene-Level Latent Variable

The active CVAE configs set `use_scene_latent: true`, so the model has one
scene-level latent for the whole graph in addition to one object-level latent
per object.

The scene-level latent is intended to capture global layout variation that is
shared by all objects in the prompt. For example, in:

```text
a dog to the left of a chair
```

the scene latent can affect the overall composition, while each object latent
can affect the dog or chair individually.

The scene posterior is produced from a graph-level readout of posterior nodes
and posterior edges:

```text
posterior_graph_state = attention_readout(posterior_nodes, posterior_edges)
scene_posterior_mu, scene_posterior_logvar =
  scene_posterior_head(posterior_graph_state)
```

`posterior_graph_state` means a single vector summarizing the whole posterior
scene graph after it has seen the ground-truth layout.

`scene_posterior_mu` and `scene_posterior_logvar` parameterize the scene-level
posterior distribution.

During training, the scene latent is sampled from this posterior:

```text
epsilon_scene ~ N(0, I)
z_scene = scene_posterior_mu + exp(0.5 * scene_posterior_logvar) * epsilon_scene
```

During inference, no ground-truth boxes are available, so the scene latent is
sampled from the standard normal prior:

```text
z_scene ~ N(0, I)
```

## Object-Level Latent Variables

The object-level latent variables capture per-object layout variation.

For every object `i`, we map its posterior node state to a Gaussian:

```text
object_posterior_mu_i, object_posterior_logvar_i = object_posterior_head(posterior_node_i)
```

`object_posterior_head` is an MLP. Its input is `posterior_node_i`, the final
posterior node state for object `i`.

`posterior_node_i` contains object label information, relation context, and
ground-truth box information because it was produced by the posterior
triple-GNN.

The raw output of this MLP is:

```text
object_posterior_stats_i = object_posterior_head(posterior_node_i)
```

`object_posterior_stats_i` means the full vector of latent distribution
parameters for object `i`.

In the current paper-facing configs, `latent_dim = 32`. If `latent_dim = 32`,
then:

```text
object_posterior_stats_i has 64 values
```

The first half becomes the mean, and the second half becomes the log variance:

```text
object_posterior_mu_i = object_posterior_stats_i[0:32]
object_posterior_logvar_i = object_posterior_stats_i[32:64]
```

`object_posterior_mu_i` is the mean of object `i`'s posterior latent
distribution.

`object_posterior_logvar_i` is the log variance of object `i`'s posterior latent
distribution.

The object latent is sampled with the reparameterization trick:

```text
epsilon_obj_i ~ N(0, I)
z_obj_i = object_posterior_mu_i + exp(0.5 * object_posterior_logvar_i) * epsilon_obj_i
```

`z_obj_i` means the sampled latent vector for object `i`.

During training, `z_obj_i` comes from the posterior. During inference, no
ground-truth boxes are available, so each object latent is sampled from the
standard normal prior:

```text
z_obj_i ~ N(0, I)
```

This is the main probabilistic part of the model. Two runs with the same scene
graph can produce different boxes because `z_obj_i` can be sampled differently.

## Decoder Triple-GNN

The decoder predicts boxes from the prior graph states and sampled latent
variables.

For each object `i`, the decoder latent context is:

```text
z_context_i = concat(z_scene, z_obj_i)
```

`z_scene` means the sampled scene-level latent vector.

`z_obj_i` means the sampled object-level latent vector for object `i`.

`z_context_i` means the latent context for object `i`.

Before the decoder node projection, the model uses FiLM-style modulation to let
the latent context scale and shift the prior node state:

```text
gamma_i, beta_i = decoder_film(z_context_i)
modulated_prior_node_i =
  prior_node_i * (1 + decoder_film_scale * tanh(gamma_i))
  + decoder_film_scale * beta_i
```

`gamma_i` and `beta_i` are the FiLM scale and shift vectors for object `i`.

`decoder_film_scale` controls how strongly the sampled latent can modulate the
graph-conditioned prior node state. Some experiments set this to `0.0`, which
removes FiLM modulation, and others set it to `1.0`.

The decoder input is:

```text
decoder_input_i = concat(modulated_prior_node_i, z_context_i)
```

`decoder_input_i` means the full input used to initialize decoder object `i`.

The decoder input is projected back to the GNN hidden dimension:

```text
decoder_node_i^0 = decoder_node_in(decoder_input_i)
```

`decoder_node_i^0` means the starting decoder node state for object `i`.

The decoder edge states start from the final prior edge states:

```text
decoder_edge_ij^0 = prior_edge_ij
```

`decoder_edge_ij^0` means the starting decoder edge state for relation `i -> j`.

The decoder then runs another triple-GNN:

```text
decoder_triple_ij = concat(decoder_node_i, decoder_edge_ij, decoder_node_j)
decoder_triple_output_ij = triple_mlp(decoder_triple_ij)
subject_msg_i, edge_update_ij, object_msg_j = split(decoder_triple_output_ij)
decoder_node_i_next = decoder_node_i + node_update(concat(decoder_node_i, aggregated_msg_i))
decoder_edge_ij_next = decoder_edge_ij + edge_update_ij
```

After the decoder triple-GNN, we have:

```text
decoder_node_i = final decoder node state for object i
```

This final decoder node state is used to predict the object's 3D box.

## Box Prediction

The box head predicts six normalized values:

```text
raw_box_i = box_head(decoder_node_i)
```

`raw_box_i` means the direct six-dimensional output from the box prediction
MLP.

The model maps it into `[0, 1]`:

```text
raw_box_i = sigmoid(raw_box_i)
```

Then the first three and last three values are sorted into min and max corners:

```text
box_min_i = min(raw_box_i[0:3], raw_box_i[3:6])
box_max_i = max(raw_box_i[0:3], raw_box_i[3:6])
pred_box_i = concat(box_min_i, box_max_i)
```

`box_min_i` means the predicted lower corner:

```text
box_min_i = [x0_i, y0_i, z0_i]
```

`box_max_i` means the predicted upper corner:

```text
box_max_i = [x1_i, y1_i, z1_i]
```

`pred_box_i` means the final predicted normalized 3D box:

```text
pred_box_i = [x0_i, y0_i, z0_i, x1_i, y1_i, z1_i]
```

For compatibility with the OSCR renderer, we also convert the min/max box into
center and size:

```text
pred_center_i = 0.5 * (box_min_i + box_max_i)
pred_size_i = box_max_i - box_min_i
```

`pred_center_i` means the center of the predicted box in `[0, 1]` coordinates.

`pred_size_i` means the predicted box width, height, and depth in normalized
coordinates.

The renderer historically expects center coordinates in `[-1, 1]`, so we also
compute:

```text
renderer_center_i = 2 * pred_center_i - 1
renderer_log_size_i = log(pred_size_i)
```

`renderer_center_i` is the center format used by the OSCR renderer.

`renderer_log_size_i` is the log-size format used by the OSCR renderer.

The training loss is still computed on `pred_box_i`, not on
`renderer_center_i`.

## Training Loss

The current 3D_SLN-style training loss is:

```text
total_loss = box_l1_loss + kl_weight * (kl_scene + kl_object)
```

`box_l1_loss` compares predicted boxes to ground-truth boxes:

```text
box_l1_loss = mean(abs(pred_box_i - gt_box_i))
```

`pred_box_i` is the predicted normalized min/max box for object `i`.

`gt_box_i` is the ground-truth normalized min/max box for object `i`.

`kl_scene` regularizes the scene-level posterior toward a standard normal:

```text
kl_scene =
  sum_over_latent_dims KL(q(z_scene | graph, gt_boxes) || N(0, I))
```

`q(z_scene | graph, gt_boxes)` means the posterior distribution for the whole
scene-level latent.

`kl_object` regularizes each object-level posterior toward a standard normal:

```text
kl_object = mean_over_valid_objects(
  sum_over_latent_dims KL(q(z_obj_i | graph, gt_box_i) || N(0, I))
)
```

`q(z_obj_i | graph, gt_box_i)` means the posterior distribution for object
`i`'s latent variable.

`valid_objects` means objects that actually exist in the batch, ignoring padded
slots.

The full training objective is:

```text
total_loss =
  mean(abs(pred_box_i - gt_box_i))
  + kl_weight * (kl_scene + kl_object)
```

In the current triple-CVAE configs:

```text
kl_weight = 0.1
```

Several runs use freebits to avoid KL collapse:

```text
kl_dim_effective = max(kl_dim - free_bits, 0)
```

`free_bits` means the KL threshold per latent dimension before the KL term starts
contributing to the loss. In the current experiment set, the important values
are:

```text
free_bits = 0.0
free_bits = 0.5
free_bits = 1.0
```

Empirically, no-freebits CVAE runs show very low spread, while freebits runs
produce much larger valid diversity.

We are currently not using orientation loss, because our data and OSCR pipeline
do not yet predict object yaw/azimuth.

We are also not using an explicit relation hinge loss in this version. The model
is expected to learn relationships through reconstructing boxes from graph
triples, matching the 3D_SLN training style more closely.

## Training vs Inference

During training:

```text
posterior sees graph + ground-truth boxes
z_scene is sampled from q(z_scene | graph, gt_boxes)
z_obj_i is sampled from q(z_obj_i | graph, gt_box_i)
decoder predicts boxes
loss compares predicted boxes to ground-truth boxes
KL pushes scene and object posterior distributions toward N(0, I)
```

During inference:

```text
only the prompt and scene graph are available
ground-truth boxes are not available
z_scene is sampled from N(0, I)
z_obj_i is sampled from N(0, I)
decoder predicts boxes from graph + sampled latents
predicted boxes are rendered into OSCR
SeeThrough3D/FLUX uses the OSCR to generate the final image
```

This gives us a probabilistic layout generator: the same prompt can produce
multiple plausible layouts by sampling different object latents.

## Deterministic GNN Baseline

The deterministic baseline uses the same object labels, relation labels, and
message-passing graph structure, but it does not use posterior encoders,
scene/object latents, KL, freebits, or stochastic sampling.

The deterministic model predicts one box per object:

```text
pred_box_i = deterministic_box_head(graph_node_i)
```

This baseline is important because it isolates the contribution of stochastic
CVAE sampling. In the latest experiments, deterministic GNNs trained on
prompt-balanced augmented data improve relation validity compared with the
original deterministic GNN, but their `Center STD`, `Size STD`, and
`Valid Diversity` remain zero by construction.

## Prompt-Balanced Augmentation

The latest augmented datasets are prompt-balanced: each unique prompt gets a
fixed number of synthetic relation-valid 3D box layouts.

The active prompt-balanced datasets are:

```text
scop_depth_crops_depth_aug_prompt250
scop_depth_crops_depth_aug_prompt400
```

These are used to test whether the model can learn a broader conditional layout
distribution than the original SCOP-Depth rows provide.

## Evaluation Summary

The current paper-facing evaluation is direct layout evaluation, not
T2I-CompBench image scoring. The evaluator compares predicted 3D boxes against
held-out SCOP-Depth metadata and writes `paper_table.md`.

The important metrics are:

```text
Rel Acc          relation correctness across all relations
2D Rel Acc       left/right/above/below/on/next-to only
3D Rel Acc       front/behind/hidden-by with depth order and projected overlap
3D Order Acc     front/behind/hidden-by depth order only
Occ. Overlap     projected overlap for 3D/occlusion relations
Box L1           reference-box coordinate error
2D/3D IoU        reference-box overlap
Center/Size STD  stochastic spread over CVAE samples
Valid Diversity  spread among samples that satisfy relations and stay in bounds
OOB              out-of-bounds or invalid box rate
Overlap          object-object 3D collision/overlap
```

The current interpretation is:

```text
relation_heuristic: sanity baseline; relation accuracy is perfect by design
deterministic augmented GNN: better relation validity, zero diversity
freebits CVAE: avoids collapse and gives meaningful valid diversity
prompt-balanced CVAE: largest valid diversity, but trades off some reference-box fidelity and relation accuracy
```
