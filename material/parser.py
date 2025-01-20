from dataclasses import dataclass, field
import bpy
import numpy as np
import gpu
import time

from .tile import get_tile_conf
from .cc import get_cc_settings
from .blender import get_blender_settings

GEO_MODE_ATTRS = [
  "g_zbuffer",
  "g_shade",
  "g_cull_front",
  "g_cull_back",
  "g_ambocclusion",
  "g_attroffset_z_enable",
  "g_attroffset_st_enable",
  "g_packed_normals",
  "g_lighttoalpha",
  "g_lighting_specular",
  "g_fresnel_color",
  "g_fresnel_alpha",
  "g_fog",
  "g_lighting",
  "g_tex_gen",
  "g_tex_gen_linear",
  "g_lod",
  "g_shade_smooth",
  "g_clipping",
]

OTHERMODE_L_ATTRS = [
  "g_mdsft_alpha_compare",
  "g_mdsft_zsrcsel",
]

OTHERMODE_H_ATTRS = [
  "g_mdsft_alpha_dither",
  "g_mdsft_rgb_dither",
  "g_mdsft_combkey",
  "g_mdsft_textconv",
  "g_mdsft_text_filt",
  "g_mdsft_textlod",
  "g_mdsft_textdetail",
  "g_mdsft_textpersp",
  "g_mdsft_cycletype",
  "g_mdsft_pipeline",
  # tlut
]

DRAW_FLAG_TEX0_MONO    = (1 << 1)
DRAW_FLAG_TEX1_MONO    = (1 << 2)
DRAW_FLAG_DECAL        = (1 << 3)
DRAW_FLAG_ALPHA_BLEND  = (1 << 4)
DRAW_FLAG_TEX0_4BIT    = (1 << 5)
DRAW_FLAG_TEX1_4BIT    = (1 << 6)
DRAW_FLAG_TEX0_3BIT    = (1 << 7)
DRAW_FLAG_TEX1_3BIT    = (1 << 8)

@dataclass
class F64Material:
    color_prim: tuple = field(default_factory=lambda: (1, 1, 1, 1))
    lod_prim: tuple = field(default_factory=lambda: (0, 0))
    prim_depth: tuple = field(default_factory=lambda: (0, 0))
    color_env: tuple = field(default_factory=lambda: (0.5, 0.5, 0.5, 0.5))
    ck: tuple = field(default_factory=lambda: (0, 0, 0, 0, 0, 0, 0, 0))
    convert: tuple = field(default_factory=lambda: (0, 0, 0, 0, 0, 0))
    color_ambient: tuple = None
    color_light: tuple= None
    
    set_prim: bool = False
    set_env: bool = False
    set_ck: bool = False
    set_convert: bool = False
    set_ambient: bool = False
    set_light: bool = False

    cc: np.ndarray = None
    blender: tuple = None
    tile_conf: np.ndarray = None
    cull: str = 'NONE'
    blend: str = 'NONE'
    depth_test: str = 'LESS_EQUAL'
    depth_write: bool = True
    flags: int = 0
    geo_mode: int = 0
    othermode_l: int = 0
    othermode_h: int = 0
    alphaClip: float = -1.0
    layer: int = 0
    tex0Buff: gpu.types.GPUTexture = None
    tex1Buff: gpu.types.GPUTexture = None


# parses a non-f3d material for the fallback renderer
def node_material_parse(mat: bpy.types.Material) -> F64Material:
  color = np.array([1, 1, 1, 1], dtype=np.float32)
  if mat.use_nodes:
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
      base_color = bsdf.inputs['Base Color'].default_value
      color = np.array([base_color[0], base_color[1], base_color[2], 1], dtype=np.float32)

  return F64Material(color, color)

# @TODO: re-use fast64 logic
def f64_parse_blend_mode(f3d_mat: any, f64mat: F64Material) -> str:
  f64mat.alphaClip = -1
  f64mat.blend = "NONE"
  is_one_cycle = f3d_mat.rdp_settings.g_mdsft_cycletype == "G_CYC_1CYCLE"

  if f3d_mat.rdp_settings.cvg_x_alpha:
    f64mat.alphaClip = 0.49
  elif (
    is_one_cycle
    and f3d_mat.rdp_settings.force_bl
    and f3d_mat.rdp_settings.blend_p1 == "G_BL_CLR_IN"
    and f3d_mat.rdp_settings.blend_a1 == "G_BL_A_IN"
    and f3d_mat.rdp_settings.blend_m1 == "G_BL_CLR_MEM"
    and f3d_mat.rdp_settings.blend_b1 == "G_BL_1MA"
  ):
    f64mat.blend = "ALPHA"
    f64mat.flags |= DRAW_FLAG_ALPHA_BLEND
  elif (
    not is_one_cycle
    and f3d_mat.rdp_settings.force_bl
    and f3d_mat.rdp_settings.blend_p2 == "G_BL_CLR_IN"
    and f3d_mat.rdp_settings.blend_a2 == "G_BL_A_IN"
    and f3d_mat.rdp_settings.blend_m2 == "G_BL_CLR_MEM"
    and f3d_mat.rdp_settings.blend_b2 == "G_BL_1MA"
  ):
    f64mat.blend = "ALPHA"
    f64mat.flags |= DRAW_FLAG_ALPHA_BLEND

def f64_material_parse(f3d_mat: any, prev_f64mat: F64Material) -> F64Material:
  from fast64_internal.utility import s_rgb_alpha_1_tuple, gammaCorrect
  from fast64_internal.f3d.f3d_material import all_combiner_uses
  cc_uses = all_combiner_uses(f3d_mat)

  f64mat = F64Material(
     color_prim = gammaCorrect(f3d_mat.prim_color),
     lod_prim   = [f3d_mat.prim_lod_min, f3d_mat.prim_lod_frac],
     prim_depth = [f3d_mat.rdp_settings.prim_depth.z, f3d_mat.rdp_settings.prim_depth.dz],
     set_prim   = f3d_mat.set_prim and cc_uses['Primitive'],
     color_env  = gammaCorrect(f3d_mat.env_color),
     set_env    = f3d_mat.set_env and cc_uses['Environment'],
     ck         = tuple((*s_rgb_alpha_1_tuple(f3d_mat.key_center), *f3d_mat.key_scale, 0)), # extra 0 for alignment
     set_ck     = f3d_mat.set_key and cc_uses['Key'],
     convert    = [getattr(f3d_mat, f"k{i}") for i in range(0, 6)],
     set_convert= f3d_mat.set_k0_5 and cc_uses['Convert'],
     color_ambient = s_rgb_alpha_1_tuple(f3d_mat.ambient_light_color),
     color_light = s_rgb_alpha_1_tuple(f3d_mat.default_light_color),
     set_ambient = f3d_mat.set_lights,
     set_light   = f3d_mat.set_lights, # shared flag

     flags = 0,
     cc = get_cc_settings(f3d_mat),
     blender= get_blender_settings(f3d_mat)
  )

  f64mat.color_prim.append(f3d_mat.prim_color[3])
  f64mat.color_env.append(f3d_mat.env_color[3])

  if f3d_mat.rdp_settings.g_cull_back: f64mat.cull = "BACK"
  if f3d_mat.rdp_settings.g_cull_front: 
    f64mat.cull = "BOTH" if f64mat.cull == "BACK" else "FRONT"
  
  f64_parse_blend_mode(f3d_mat, f64mat)

  from fast64_internal.f3d.f3d_gbi import get_F3D_GBI
  from fast64_internal.f3d.f3d_material import get_textlut_mode
  gbi = get_F3D_GBI()
  geo_mode = othermode_l = othermode_h = 0
  # TODO: use geo_modes_in_ucode (T3D UI pr) to check if the geo mode exists in the current ucode
  for i, attr in enumerate(GEO_MODE_ATTRS):
    if not getattr(gbi, attr.upper().replace("G_TEX_GEN", "G_TEXTURE_GEN").replace("G_SHADE_SMOOTH", "G_SHADING_SMOOTH"), False):
      continue
    geo_mode |= int(getattr(f3d_mat.rdp_settings, attr)) << i
  for i, attr in enumerate(OTHERMODE_L_ATTRS):
    othermode_l |= getattr(gbi, getattr(f3d_mat.rdp_settings, attr))
  for i, attr in enumerate(OTHERMODE_H_ATTRS):
    othermode_h |= getattr(gbi, getattr(f3d_mat.rdp_settings, attr))
  othermode_h |= getattr(gbi, get_textlut_mode(f3d_mat))
  f64mat.geo_mode, f64mat.othermode_l, f64mat.othermode_h = geo_mode, othermode_l, othermode_h
  
  game_mode = bpy.context.scene.gameEditorMode
  f64mat.layer = getattr(f3d_mat.draw_layer, game_mode.lower(), None)

  if f3d_mat.rdp_settings.zmode == 'ZMODE_DEC':
    f64mat.depth_test = 'EQUAL'
    f64mat.flags |= DRAW_FLAG_DECAL

  f64mat.depth_write = f3d_mat.rdp_settings.z_upd

  # Note: doing 'gpu.texture.from_image' seems to cost nothing, caching is not needed
  if f3d_mat.tex0.tex_set:
    if f3d_mat.tex0.tex:
      f64mat.tex0Buff = gpu.texture.from_image(f3d_mat.tex0.tex)
      if f3d_mat.tex0.tex_format == 'I4' or f3d_mat.tex0.tex_format == 'I8':
        f64mat.flags |= DRAW_FLAG_TEX0_MONO
      if f3d_mat.tex0.tex_format == 'I4' or f3d_mat.tex0.tex_format == 'IA8':
        f64mat.flags |= DRAW_FLAG_TEX0_4BIT
      if f3d_mat.tex0.tex_format == 'IA4':
        f64mat.flags |= DRAW_FLAG_TEX0_3BIT
    else:
      f64mat.tex0Buff = gpu.texture.from_image(bpy.data.images["f64render_missing_texture"])
      f64mat.flags |= DRAW_FLAG_TEX0_MONO

  if f3d_mat.tex1.tex_set:
    if f3d_mat.tex1.tex:
      f64mat.tex1Buff = gpu.texture.from_image(f3d_mat.tex1.tex)
      if f3d_mat.tex1.tex_format == 'I4' or f3d_mat.tex1.tex_format == 'I8':
        f64mat.flags |= DRAW_FLAG_TEX1_MONO
      if f3d_mat.tex1.tex_format == 'I4' or f3d_mat.tex1.tex_format == 'IA8':
        f64mat.flags |= DRAW_FLAG_TEX1_4BIT
      if f3d_mat.tex1.tex_format == 'IA4':
        f64mat.flags |= DRAW_FLAG_TEX1_3BIT
    else:
      f64mat.tex1Buff = gpu.texture.from_image(bpy.data.images["f64render_missing_texture"])
      f64mat.flags |= DRAW_FLAG_TEX1_MONO


  f64mat.tile_conf = get_tile_conf(f3d_mat)

  return f64mat