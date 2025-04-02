import dataclasses, struct
import numpy as np

import bpy
import mathutils
import gpu

from .material.parser import (
  parse_f3d_rendermode_preset,
  quantize_direction,
  quantize_srgb,
  F64Material,
  F64RenderState,
  F64Rendermode,
  F64Light,
  quantize_tuple,
)
from .material.cc import SOLID_CC
from .material.tile import get_tile_conf
from .mesh.mesh import MeshBuffers
from .properties import F64RenderSettings

LIGHT_STRUCT = "4f 3f 4x"           # color, direction, padding
TILE_STRUCT = "2f 2f 2f 2f i 12x"    # mask, shift, low, high, padding, flags

UNIFORM_BUFFER_STRUCT = struct.Struct(
  (TILE_STRUCT * 8) +               # texture configurations
  (LIGHT_STRUCT * 8) +              # lights
  "8i"                              # blender
  "16i"                             # color-combiner settings
  "i i i i"                         # geoMode, other-low, other-high, flags
  "4f 4f 4f 4f"                     # prim, prim_lod, prim-depth, env, ambient
  "3f f 3f i 3f i"                  # ck center, alpha clip, ck scale, light count, width, uv basis
  "6f i 4x"                         # k0-k5, mipmap count, padding
)

@dataclasses.dataclass
class ObjRenderInfo:
  obj: bpy.types.Object
  mvp_matrix: mathutils.Matrix
  normal_matrix: mathutils.Matrix
  render_obj: MeshBuffers
  mats: list[tuple[int, int, F64Material]] # mat idx, indice count, material

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
    ck=tuple((*quantize_srgb(f64render_rs.default_key_center, False), *f64render_rs.default_key_scale, *f64render_rs.default_key_width)),
    convert=quantize_tuple(f64render_rs.default_convert, 9.0, -1.0, 1.0),
    cc=SOLID_CC,
    alpha_clip=-1,
    render_mode=F64Rendermode(),
    tex_confs=([get_tile_conf(getattr(f64render_rs, f"default_tex{i}")) for i in range(0, 8)]),
  )
  state.lights[0] = F64Light(quantize_srgb(fast64_rs.light0Color, force_alpha=True), quantize_direction(fast64_rs.light0Direction))
  state.lights[1] = F64Light(quantize_srgb(fast64_rs.light1Color, force_alpha=True), quantize_direction(fast64_rs.light1Direction))
  state.set_from_rendermode(parse_f3d_rendermode_preset("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"))
  return state

def draw_f64_obj(render_engine: "Fast64RenderEngine", render_state: F64RenderState, info: ObjRenderInfo):
  mvp = np.array(info.mvp_matrix)
  bbox = (mvp @ info.render_obj.bound_box.T).T  # apply view and projection
  bbox = bbox[:, :3] / bbox[:, 3, None]  # perspective divide

  # check if any orientation (so [:, :3]) of all corners is fully outside the -1 to 1 range
  if (np.all(bbox[:, 0] < -1) or np.all(bbox[:, 0] > 1) or
    np.all(bbox[:, 1] < -1) or np.all(bbox[:, 1] > 1) or
    np.all(bbox[:, 2] < -1) or np.all(bbox[:, 2] > 1)):
    if not info.obj.use_f3d_culling: # if obj is not meant to be culled in game, apply all materials
      for mat_idx, indices_count, f64mat in info.mats: render_state.set_if_not_none(f64mat.state)
    return

  render_engine.shader.uniform_float("matMVP", info.mvp_matrix)
  render_engine.shader.uniform_float("matNorm", info.normal_matrix)

  for mat_idx, indices_count, f64mat in info.mats:
    render_state.set_if_not_none(f64mat.state)

    gpu.state.face_culling_set(f64mat.cull)
    if not render_engine.shader_info_img_impl:
      gpu.state.blend_set(render_state.render_mode.blend)
      gpu.state.depth_test_set(render_state.render_mode.depth_test)
      gpu.state.depth_mask_set(render_state.render_mode.depth_write)

    for i in range(8):
      if render_state.tex_confs[i].buff is not render_engine.last_used_textures.get(i): 
        render_engine.shader.uniform_sampler(f"tex{i}", render_state.tex_confs[i].buff)
        render_engine.last_used_textures[i] = render_state.tex_confs[i].buff

    light_data = []
    for l in render_state.lights[:render_state.light_count]:
      light_data.extend(l.color)
      light_data.extend(l.direction)
    light_data.extend([0.0] * ((8 - render_state.light_count) * 7))

    tex_data = []
    for t in render_state.tex_confs: tex_data.extend(t.values)

    info.render_obj.mat_data[mat_idx] = UNIFORM_BUFFER_STRUCT.pack(
      *tex_data,
      *light_data,
      *render_state.render_mode.blender,
      *render_state.cc,
      f64mat.geo_mode,
      f64mat.othermode_l,
      f64mat.othermode_h,
      f64mat.flags | render_state.render_mode.flags,
      *render_state.prim_color,
      *render_state.prim_lod,
      *f64mat.prim_depth,
      *render_state.env_color,
      *render_state.ambient_color,
      *render_state.ck[:3],
      render_state.alpha_clip,
      *render_state.ck[3:6],
      render_state.light_count,
      *render_state.ck[6:9],
      f64mat.uv_basis,
      *render_state.convert,
      f64mat.mip_count,
    )
    
    info.render_obj.ubo_mat_data[mat_idx].update(info.render_obj.mat_data[mat_idx])                        
    render_engine.shader.uniform_block("material", info.render_obj.ubo_mat_data[mat_idx])

    if render_engine.draw_range_impl:
      info.render_obj.batch.draw_range(render_engine.shader, elem_start=info.render_obj.index_offsets[mat_idx] * 3, elem_count=indices_count)
    else:
      info.render_obj.batch[mat_idx].draw(render_engine.shader)