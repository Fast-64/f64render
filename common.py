import dataclasses
import struct
import typing
import numpy as np

import bpy
import mathutils
import gpu

from .material.parser import (
    parse_f3d_rendermode_preset,
    quantize_direction,
    quantize_srgb,
    quantize_tuple,
    f64_material_parse,
    node_material_parse,
    UNIFORM_BUFFER_STRUCT,
    F64Material,
    F64RenderState,
    F64Light,
)
from .material.cc import SOLID_CC
from .material.tile import get_tile_conf
from .mesh.mesh import MeshBuffers, mesh_to_buffers
from .mesh.gpu_batch import batch_for_shader, create_vert_buf
from .properties import F64RenderSettings
from .globals import F64_GLOBALS

if typing.TYPE_CHECKING:
    from .renderer import Fast64RenderEngine

FALLBACK_MATERIAL = F64Material(state=F64RenderState(cc=SOLID_CC))
FALLBACK_MATERIAL.state.save_cache()


def get_struct_ubo_size(s: struct.Struct):
    return (s.size + 15) & ~15  # force 16-byte alignment


UBO_SIZE = get_struct_ubo_size(UNIFORM_BUFFER_STRUCT)


@dataclasses.dataclass
class ObjRenderInfo:
    obj: bpy.types.Object
    mvp_matrix: mathutils.Matrix
    normal_matrix: mathutils.Matrix
    render_obj: MeshBuffers
    mats: list[tuple[int, int, F64Material]]  # mat idx, indice count, material


def get_scene_render_state(scene: bpy.types.Scene):
    fast64_rs = scene.fast64.renderSettings
    f64render_rs: F64RenderSettings = scene.f64render.render_settings
    state = F64RenderState(
        lights=[F64Light(direction=(0, 0, 0)) for _x in range(0, 8)],
        ambient_color=quantize_srgb(fast64_rs.ambientColor, force_alpha=True),
        light_count=2,
        prim_color=quantize_srgb(f64render_rs.default_prim_color),
        prim_lod=(f64render_rs.default_lod_frac, f64render_rs.default_lod_min),
        env_color=quantize_srgb(f64render_rs.default_env_color),
        ck=tuple(
            (
                *quantize_srgb(f64render_rs.default_key_center, False),
                *f64render_rs.default_key_scale,
                *f64render_rs.default_key_width,
            )
        ),
        convert=quantize_tuple(f64render_rs.default_convert, 9.0, -1.0, 1.0),
        cc=SOLID_CC,
        tex_confs=([get_tile_conf(getattr(f64render_rs, f"default_tex{i}")) for i in range(0, 8)]),
    )
    state.lights[0] = F64Light(
        quantize_srgb(fast64_rs.light0Color, force_alpha=True), quantize_direction(fast64_rs.light0Direction)
    )
    state.lights[1] = F64Light(
        quantize_srgb(fast64_rs.light1Color, force_alpha=True), quantize_direction(fast64_rs.light1Direction)
    )
    state.set_from_rendermode(parse_f3d_rendermode_preset("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"))
    state.save_cache()
    return state


def draw_f64_obj(render_engine: "Fast64RenderEngine", render_state: F64RenderState, info: ObjRenderInfo):
    mvp = info.mvp_matrix
    bbox = info.render_obj.bounding_box

    # we need to figure out where what the min max range is for all axis, but we need to do this after projection
    # would've been a neat optimization but it does not work :(
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    for v in bbox:
        v = mvp @ v
        x, y, z = v.xyz / v.w

        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        if z < min_z:
            min_z = z
        if z > max_z:
            max_z = z

    if (max_x < -1 or min_x > 1) or (max_y < -1 or min_y > 1) or (max_z < -1 or min_z > 1):
        if not info.obj.use_f3d_culling:
            for _, _, f64mat in info.mats:
                render_state.set_values_from_cache(f64mat.state)
        return

    render_engine.shader.uniform_float("matMVP", info.mvp_matrix)
    render_engine.shader.uniform_float("matNorm", info.normal_matrix)

    for mat_idx, indices_count, f64mat in info.mats:
        render_state.set_values_from_cache(f64mat.state)

        gpu.state.face_culling_set(f64mat.cull)
        if not render_engine.use_atomic_rendering:
            gpu.state.blend_set(render_state.render_mode.blend)
            gpu.state.depth_test_set(render_state.render_mode.depth_test)
            gpu.state.depth_mask_set(render_state.render_mode.depth_write)

        for i in range(8):
            if render_state.tex_confs[i].buff is not render_engine.last_used_textures.get(i):
                render_engine.shader.uniform_sampler(f"tex{i}", render_state.tex_confs[i].buff)
                render_engine.last_used_textures[i] = render_state.tex_confs[i].buff

        info.render_obj.ubo_mat_data[mat_idx].update(render_state.cached_values)
        render_engine.shader.uniform_block("material", info.render_obj.ubo_mat_data[mat_idx])

        if render_engine.draw_range_impl:
            info.render_obj.batch.draw_range(
                render_engine.shader, elem_start=info.render_obj.index_offsets[mat_idx] * 3, elem_count=indices_count
            )
        else:
            info.render_obj.batch[mat_idx].draw(render_engine.shader)


def collect_obj_info(
    render_engine: "Fast64RenderEngine",
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
    hidden_objs_names: set[str],
    space_view_3d: bpy.types.SpaceView3D,
    projection_matrix: mathutils.Matrix,
    view_matrix: mathutils.Matrix,
    always_set: bool,
    set_light_dir=True,
):
    if (
        obj.name in hidden_objs_names
        or obj.type not in {"MESH", "CURVE", "SURFACE", "FONT"}
        or obj.data is None
        or (space_view_3d.local_view and not obj.local_view_get(space_view_3d))
    ):
        return
    mesh_id = f"{obj.name}#{obj.data.name}"
    if mesh_id in F64_GLOBALS.meshCache:  # check for objects that transitioned from non-f3d to f3d materials
        render_obj = F64_GLOBALS.meshCache[mesh_id]
    else:  # Mesh not cached: parse & convert mesh data, then prepare a GPU batch
        if obj.mode == "EDIT":
            mesh = obj.evaluated_get(depsgraph).to_mesh()
        else:
            mesh = obj.evaluated_get(depsgraph).to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

        render_obj = F64_GLOBALS.meshCache[mesh_id] = mesh_to_buffers(mesh)
        render_obj.mesh_name = obj.data.name
        render_obj.bounding_box = [mathutils.Vector((*corner, 1)) for corner in obj.bound_box]

        mat_count = max(len(obj.material_slots), 1)
        vert_buf = create_vert_buf(
            render_engine.vbo_format,
            render_obj.vert,
            render_obj.norm,
            render_obj.color,
            render_obj.uv,
        )
        if render_engine.draw_range_impl:
            render_obj.batch = batch_for_shader(vert_buf, render_obj.indices)
        else:  # we need to create batches for each material
            if not obj.material_slots:  # if no material slot, we only have one batch for the whole geo
                render_obj.batch = [batch_for_shader(vert_buf, render_obj.indices)]
            else:
                render_obj.batch = []
            for i, slot in enumerate(obj.material_slots):
                indices = render_obj.indices[render_obj.index_offsets[i] : render_obj.index_offsets[i + 1]]
                if len(indices) == 0:  # ignore unused materials
                    render_obj.batch.append(None)
                else:
                    render_obj.batch.append(batch_for_shader(vert_buf, indices))

        render_obj.ubo_mat_data = [None] * mat_count

        for i in range(mat_count):
            render_obj.ubo_mat_data[i] = gpu.types.GPUUniformBuf(bytes(UBO_SIZE))

        obj.to_mesh_clear()

    modelview_matrix = obj.matrix_world
    mvp_matrix = projection_matrix @ modelview_matrix  # could we use numpy?
    normal_matrix = (view_matrix @ obj.matrix_world).to_3x3().inverted().transposed()

    info = ObjRenderInfo(obj, mvp_matrix, normal_matrix, render_obj, [])

    if len(obj.material_slots) == 0:  # fallback if no material, f3d or otherwise
        info.mats.append((0, len(render_obj.indices) * 3, FALLBACK_MATERIAL))
    for i, slot in enumerate(obj.material_slots):
        indices_count = (render_obj.index_offsets[i + 1] - render_obj.index_offsets[i]) * 3
        if indices_count == 0:  # ignore unused materials
            continue
        if slot.material is None:
            continue

        if slot.material not in F64_GLOBALS.materials_cache:
            try:
                if slot.material.is_f3d:
                    F64_GLOBALS.materials_cache[slot.material] = f64_material_parse(
                        slot.material.f3d_mat, always_set, set_light_dir
                    )
                else:  # fallback
                    F64_GLOBALS.materials_cache[slot.material] = node_material_parse(slot.material)
            except Exception as e:
                print(f'Error parsing material "{slot.material.name}": {e}')
                F64_GLOBALS.materials_cache[slot.material] = FALLBACK_MATERIAL

        f64mat = F64_GLOBALS.materials_cache[slot.material]
        if f64mat.cull == "BOTH":
            continue

        info.mats.append((i, indices_count, f64mat))
    return info
