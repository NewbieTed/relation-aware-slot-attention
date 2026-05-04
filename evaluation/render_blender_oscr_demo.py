"""Render SeeThrough3D-style Blender OSCR demos from saved GNN records.

The script consumes ``top_left_front_oscr_records.json`` style files produced by
``evaluation.demo_top_left_front_oscr``. It does not run the GNN. Instead, it
maps predicted normalized centers/sizes into a small Blender scene and renders
very transparent cuboids similar to SeeThrough3D's OSCR condition images.

This is a diagnostic renderer for exploring OSCR visual design locally. It is
not used by training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


OBJECT_COLORS = [
    (0.00, 0.78, 1.00),
    (1.00, 0.12, 0.36),
    (0.20, 0.85, 0.35),
    (1.00, 0.75, 0.10),
    (0.62, 0.35, 1.00),
    (1.00, 0.45, 0.05),
]

RGB_FACE_COLORS = [
    # SeeThrough3D's get_primitive_object_translucent_rgb builds:
    # green faces for the first four cube polygons, blue for the fifth, red
    # for the sixth, then reorders to [blue, red, green, green, green, green].
    # Keep this order rather than guessing Blender's world-space face normals.
    (0.0, 0.0, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.5, 0.0),
]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Blender-based OSCR demos from saved GNN JSON records.")
    parser.add_argument("--records-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--world-scale", type=float, default=3.5)
    parser.add_argument("--depth-scale", type=float, default=3.0)
    parser.add_argument("--face-alpha", type=float, default=0.025)
    parser.add_argument(
        "--face-alpha-scale",
        type=float,
        default=1.0,
        help="Debug brightness multiplier applied to the SeeThrough-style face alpha.",
    )
    parser.add_argument("--edge-radius", type=float, default=0.012)
    parser.add_argument("--orthographic-scale", type=float, default=7.0)
    parser.add_argument("--camera-x", type=float, default=4.2)
    parser.add_argument("--camera-y", type=float, default=-7.0)
    parser.add_argument("--camera-z", type=float, default=4.3)
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--rgb-faces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edges", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ground", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shadows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--transparent-background", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--background", choices=("black", "white"), default="white")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _script_argv() -> list[str] | None:
    """Return arguments intended for this script when launched by Blender.

    ``blender --background --python script.py -- --flag value`` keeps Blender's
    own arguments in ``sys.argv``. Standard argparse would try to parse all of
    them, so we explicitly keep only the portion after ``--``. Some Blender
    builds strip the marker, so we also support finding the first known script
    flag as a fallback.
    """

    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    known_flags = {"--records-json", "--output-dir"}
    for index, value in enumerate(sys.argv):
        if value in known_flags:
            return sys.argv[index:]
    return None


def _safe_name(prompt: str, index: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in prompt).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return f"{index:03d}_{safe[:80] or 'prompt'}"


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.curves):
        if block.users == 0:
            bpy.data.curves.remove(block)


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _make_face_material(
    name: str,
    color: tuple[float, float, float],
    alpha: float,
    *,
    shadow: bool = False,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = False
    mat.use_screen_refraction = False
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    mat.show_transparent_back = False
    if not shadow:
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    display_alpha = max(0.0, min(1.0, alpha))
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, display_alpha)
        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Roughness"].default_value = 0.72
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = (*color, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = 0.8
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = False
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = False
    return mat


def _make_emission_material(name: str, color: tuple[float, float, float], alpha: float = 1.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, alpha)
    emission.inputs["Strength"].default_value = 1.5
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _create_edge(start: Vector, end: Vector, *, radius: float, material: bpy.types.Material, name: str) -> None:
    midpoint = (start + end) * 0.5
    direction = end - start
    length = direction.length
    if length <= 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=length, location=midpoint)
    edge = bpy.context.object
    edge.name = name
    edge.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    edge.data.materials.append(material)


def _draw_edges(center: Vector, dims: Vector, *, radius: float, material: bpy.types.Material, name: str) -> None:
    hx, hy, hz = dims.x * 0.5, dims.y * 0.5, dims.z * 0.5
    corners: dict[tuple[int, int, int], Vector] = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners[(sx, sy, sz)] = center + Vector((sx * hx, sy * hy, sz * hz))
    edge_pairs = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            edge_pairs.append((corners[(-1, sy, sz)], corners[(1, sy, sz)]))
    for sx in (-1, 1):
        for sz in (-1, 1):
            edge_pairs.append((corners[(sx, -1, sz)], corners[(sx, 1, sz)]))
    for sx in (-1, 1):
        for sy in (-1, 1):
            edge_pairs.append((corners[(sx, sy, -1)], corners[(sx, sy, 1)]))
    for edge_index, (start, end) in enumerate(edge_pairs):
        _create_edge(start, end, radius=radius, material=material, name=f"{name}_edge_{edge_index:02d}")


def _add_label(text: str, location: Vector, *, color: tuple[float, float, float], size: float = 0.12) -> None:
    bpy.ops.object.text_add(location=location)
    label = bpy.context.object
    label.name = f"label_{text}"
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.data.size = size
    label.data.materials.append(_make_emission_material(f"label_mat_{text}", color, 1.0))
    label.rotation_euler = bpy.context.scene.camera.rotation_euler


def _to_world(
    center_xyz: list[float],
    size_xyz: list[float],
    *,
    world_scale: float,
    depth_scale: float,
) -> tuple[Vector, Vector]:
    # Predicted x/y are normalized image-plane coordinates. Predicted z is depth.
    # Blender uses z as vertical, so image y is inverted into world z.
    x, y, z = center_xyz
    sx, sy, sz = size_xyz
    dims = Vector(
        (
            max(0.12, sx * world_scale),
            max(0.12, sz * depth_scale),
            max(0.12, sy * world_scale),
        )
    )
    center = Vector(
        (
            x * world_scale,
            -z * depth_scale,
            -y * world_scale + dims.z * 0.5,
        )
    )
    return center, dims


def _setup_scene(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES" if args.engine == "cycles" else "BLENDER_EEVEE"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = False
    else:
        scene.eevee.use_gtao = False
        scene.eevee.use_bloom = False
        scene.eevee.use_soft_shadows = bool(args.shadows)
        scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x = args.image_size
    scene.render.resolution_y = args.image_size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(args.transparent_background)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = bpy.data.worlds.new("OSCRWorld")
    scene.world.color = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)

    bpy.ops.object.light_add(type="AREA", location=(0, -4.0, 5.5))
    light = bpy.context.object
    light.name = "softbox"
    light.data.energy = 160 if scene.render.engine == "CYCLES" else 250
    light.data.size = 5.0
    light.data.use_shadow = bool(args.shadows)

    if args.ground:
        bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0, 0, 0))
        plane = bpy.context.object
        plane.name = "ground_plane"
        plane.visible_shadow = bool(args.shadows)
        plane.data.materials.append(_make_face_material("ground_mat", (0.08, 0.08, 0.08), 0.12))

    bpy.ops.object.camera_add(location=(args.camera_x, args.camera_y, args.camera_z))
    camera = bpy.context.object
    camera.name = "OSCRCamera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = args.orthographic_scale
    _look_at(camera, Vector((0, 0, 0.7)))
    scene.camera = camera


def _render_record(record: dict[str, Any], *, args: argparse.Namespace, index: int) -> dict[str, Any]:
    _clear_scene()
    _setup_scene(args)
    labels = record["labels"]
    centers = record["predicted_centers"]
    sizes = record["predicted_sizes"]
    edge_summaries = []

    # Draw farther boxes first so transparent faces accumulate in a stable order.
    order = sorted(range(len(labels)), key=lambda idx: centers[idx][2])
    for draw_order, slot_index in enumerate(order):
        label = labels[slot_index]
        object_color = OBJECT_COLORS[slot_index % len(OBJECT_COLORS)]
        center, dims = _to_world(
            centers[slot_index],
            sizes[slot_index],
            world_scale=args.world_scale,
            depth_scale=args.depth_scale,
        )
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
        cube = bpy.context.object
        cube.name = f"cuboid_{slot_index}_{label}"
        cube.dimensions = dims
        bpy.context.view_layer.update()
        cube.visible_shadow = bool(args.shadows)
        face_colors = RGB_FACE_COLORS if args.rgb_faces else [object_color] * 6
        face_alpha = args.face_alpha * args.face_alpha_scale
        for face_index, color in enumerate(face_colors):
            cube.data.materials.append(
                _make_face_material(
                    f"face_mat_{slot_index}_{label}_{face_index}",
                    color,
                    face_alpha,
                    shadow=args.shadows,
                )
            )
        if len(cube.data.polygons) == 6:
            for face_index, polygon in enumerate(cube.data.polygons):
                polygon.material_index = face_index
        if args.edges:
            edge_mat = _make_emission_material(f"edge_mat_{slot_index}_{label}", object_color, 1.0)
            _draw_edges(center, dims, radius=args.edge_radius, material=edge_mat, name=f"cuboid_{slot_index}_{label}")
        if args.labels:
            _add_label(
                label,
                center + Vector((0, -dims.y * 0.65, dims.z * 0.65 + 0.12 + draw_order * 0.03)),
                color=object_color,
            )
        edge_summaries.append(
            {
                "label": label,
                "world_center": [center.x, center.y, center.z],
                "world_dims": [dims.x, dims.y, dims.z],
            }
        )

    stem = _safe_name(record["prompt"], index)
    output_path = args.output_dir / f"{stem}_blender_oscr.png"
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return {
        "prompt": record["prompt"],
        "output": str(output_path),
        "objects": edge_summaries,
    }


def main() -> int:
    args = make_parser().parse_args(_script_argv())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.records_json.read_text())
    if args.limit is not None:
        records = records[: args.limit]

    manifest = []
    for index, record in enumerate(records):
        result = _render_record(record, args=args, index=index)
        manifest.append(result)
        print(f"Rendered {result['output']}")

    (args.output_dir / "blender_oscr_manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
