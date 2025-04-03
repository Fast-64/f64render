import copy
from typing import NamedTuple

import bpy
import numpy as np

from .material.parser import parse_f3d_rendermode_preset, F64RenderState
from .common import ObjRenderInfo, draw_f64_obj, get_scene_render_state
from .properties import F64RenderSettings
from .globals import F64_GLOBALS

class AreaRenderInfo(NamedTuple): # areas, etc
  render_state: F64RenderState
  name: str
  def __hash__(self):
    return hash(self.name)

def get_sm64_area_childrens(scene: bpy.types.Scene):
  global F64_GLOBALS
  area_objs = []
  def get_children_until_next_area(obj: bpy.types.Object):
    children = []
    for child in sorted(obj.children, key=lambda item: item.name): 
      if child not in area_objs: 
        children.append(child)
        children.extend(get_children_until_next_area(child))
    return children

  if F64_GLOBALS.area_lookup is not None:
    return F64_GLOBALS.area_lookup

  area_lookup = {}
  for obj in bpy.data.objects: # find all area type objects
    if obj.sm64_obj_type == "Area Root": area_objs.append(obj)

  render_state = get_scene_render_state(scene)
  for area_obj in area_objs:
    area_info = AreaRenderInfo(render_state, area_obj.name)
    for child in get_children_until_next_area(area_obj):
      area_lookup[child.name] = area_info

  fake_area = AreaRenderInfo(render_state, "")
  for obj in bpy.data.objects:
    if obj.name not in area_lookup:
      area_lookup[obj.name] = fake_area

  F64_GLOBALS.area_lookup = area_lookup
  return area_lookup

# TODO if porting to fast64, reuse existing default layer dict
SM64_DEFAULT_LAYERS = (("G_RM_ZB_OPA_SURF", "G_RM_ZB_OPA_SURF2"), 
                      ("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"), 
                      ("G_RM_AA_ZB_OPA_DECAL", "G_RM_AA_ZB_OPA_DECAL2"), 
                      ("G_RM_AA_ZB_OPA_INTER", "G_RM_AA_ZB_OPA_INTER2"), 
                      ("G_RM_AA_ZB_TEX_EDGE", "G_RM_AA_ZB_TEX_EDGE2"), 
                      ("G_RM_AA_ZB_XLU_SURF", "G_RM_AA_ZB_XLU_SURF2"), 
                      ("G_RM_AA_ZB_XLU_DECAL", "G_RM_AA_ZB_XLU_DECAL2"), 
                      ("G_RM_AA_ZB_XLU_INTER", "G_RM_AA_ZB_XLU_INTER2"))

def draw_sm64_scene(render_engine: "Fast64RenderEngine", depsgraph: bpy.types.Depsgraph, objs_info: list[ObjRenderInfo]):
  f64render_rs: F64RenderSettings = depsgraph.scene.f64render.render_settings

  layer_rendermodes = {} # TODO: should this be cached globally?
  world = depsgraph.scene.world
  for layer, (cycle1, cycle2) in enumerate(SM64_DEFAULT_LAYERS):
    if world:
      cycle1, cycle2 = (getattr(world, f"draw_layer_{layer}_cycle_{cycle}") for cycle in range(1, 3))
    layer_rendermodes[layer] = parse_f3d_rendermode_preset(cycle1, cycle2)

  render_type = f64render_rs.sm64_render_type
  ignore, collision = render_type == "IGNORE", render_type == "COLLISION"
  area_lookup = get_sm64_area_childrens(depsgraph.scene)
  area_queue: dict[AreaRenderInfo, dict[int, dict[str, ObjRenderInfo]]] = {}
  for info in objs_info:
    if (ignore and info.obj.ignore_render) or collision and info.obj.ignore_collision:
      continue
    name = info.obj.name
    area = area_lookup[name]
    layer_queue = area_queue.setdefault(area, {}) # if area has no queue, create it
    for mat_info in info.mats:
      mat = mat_info[2]
      obj_queue = layer_queue.setdefault(int(mat.layer), {}) # if layer has no queue, create it
      if name not in obj_queue: # if obj not already present in the layer's obj queue, create a shallow copy
        obj_info = obj_queue[name] = copy.copy(info)
        obj_info.mats = []
      obj_queue[name].mats.append(mat_info)

  for area, layer_queue in area_queue.items():
    render_state = area.render_state.copy()
    for layer, obj_queue in sorted(layer_queue.items(), key=lambda item: item[0]): # sort by layer
      render_state.set_from_rendermode(layer_rendermodes[layer])
      for info in dict(sorted(obj_queue.items(), key=lambda item: item[0])): # sort by obj name
        draw_f64_obj(render_engine, render_state, obj_queue[info])