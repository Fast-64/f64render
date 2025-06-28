import copy
from typing import NamedTuple

import bpy
import mathutils

from .material.parser import parse_f3d_rendermode_preset, F64RenderState
from .common import ObjRenderInfo, draw_f64_obj, collect_obj_info, get_scene_render_state
from .properties import F64RenderSettings
from .globals import F64_GLOBALS


class AreaRenderInfo(NamedTuple):  # areas, etc
    render_state: F64RenderState
    name: str

    def __hash__(self):
        return hash(self.name)


def get_sm64_area_childrens(scene: bpy.types.Scene):
    if F64_GLOBALS.sm64_area_lookup is not None:
        return F64_GLOBALS.sm64_area_lookup

    sm64_area_lookup = {}
    render_state = get_scene_render_state(scene)
    level_objs: list[bpy.types.Object] = []
    area_objs: list[bpy.types.Object] = []

    def get_area_children(obj: bpy.types.Object, name: str = ""):
        for child in sorted(obj.children, key=lambda item: item.name):
            if child not in area_objs:
                sm64_area_lookup[child.name] = AreaRenderInfo(render_state, name)
                get_area_children(child, name)
            else:
                get_area_children(child, child.name)

    def get_level_children(obj: bpy.types.Object, name: str):
        for child in sorted(obj.children, key=lambda item: item.name):
            if child in area_objs:
                get_area_children(child, child.name)
            else:
                sm64_area_lookup[child.name] = AreaRenderInfo(render_state, name)
                get_level_children(child, name)

    for obj in bpy.data.objects:  # find all area type objects
        if obj.type == "EMPTY" and obj.sm64_obj_type == "Area Root":
            area_objs.append(obj)
        if obj.type == "EMPTY":
            if obj.sm64_obj_type == "Level Root":
                level_objs.append(obj)
            if obj.sm64_obj_type == "Area Root":
                area_objs.append(obj)

    for level_obj in level_objs:
        get_level_children(level_obj, level_obj.name)

    fake_area = AreaRenderInfo(render_state, "")
    for obj in bpy.data.objects:
        if obj.name not in sm64_area_lookup:
            sm64_area_lookup[obj.name] = fake_area

    F64_GLOBALS.sm64_area_lookup = sm64_area_lookup
    return sm64_area_lookup


# TODO if porting to fast64, reuse existing default layer dict
DEFAULT_LAYERS = (
    ("G_RM_ZB_OPA_SURF", "G_RM_ZB_OPA_SURF2"),
    ("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"),
    ("G_RM_AA_ZB_OPA_DECAL", "G_RM_AA_ZB_OPA_DECAL2"),
    ("G_RM_AA_ZB_OPA_INTER", "G_RM_AA_ZB_OPA_INTER2"),
    ("G_RM_AA_ZB_TEX_EDGE", "G_RM_AA_ZB_TEX_EDGE2"),
    ("G_RM_AA_ZB_XLU_SURF", "G_RM_AA_ZB_XLU_SURF2"),
    ("G_RM_AA_ZB_XLU_DECAL", "G_RM_AA_ZB_XLU_DECAL2"),
    ("G_RM_AA_ZB_XLU_INTER", "G_RM_AA_ZB_XLU_INTER2"),
)


def draw_sm64_scene(
    render_engine: "Fast64RenderEngine",
    depsgraph: bpy.types.Depsgraph,
    hidden_objs_names: set[str],
    space_view_3d: bpy.types.SpaceView3D,
    projection_matrix: mathutils.Matrix,
    view_matrix: mathutils.Matrix,
    always_set: bool,
):
    f64render_rs: F64RenderSettings = depsgraph.scene.f64render.render_settings

    layer_rendermodes = {}  # TODO: should this be cached globally?
    world = depsgraph.scene.world
    for layer, (cycle1, cycle2) in enumerate(DEFAULT_LAYERS):
        if world:
            cycle1, cycle2 = (getattr(world, f"draw_layer_{layer}_cycle_{cycle}") for cycle in (1, 2))
        rm_state = F64RenderState()
        rm_state.set_from_rendermode(parse_f3d_rendermode_preset(cycle1, cycle2))
        rm_state.save_cache()
        layer_rendermodes[layer] = rm_state

    ignore, collision = f64render_rs.render_type == "IGNORE", f64render_rs.render_type == "COLLISION"
    specific_area = f64render_rs.sm64_specific_area.name if f64render_rs.sm64_specific_area else None
    area_lookup = get_sm64_area_childrens(depsgraph.scene)
    area_queue: dict[AreaRenderInfo, dict[int, dict[str, ObjRenderInfo]]] = {}

    for obj in depsgraph.objects:
        obj_name = obj.name
        area = area_lookup[obj_name]
        if (
            (ignore and obj.ignore_render)
            or (collision and obj.ignore_collision)
            or (specific_area and area.name != specific_area)
        ):
            continue
        obj_info = collect_obj_info(
            render_engine, obj, depsgraph, hidden_objs_names, space_view_3d, projection_matrix, view_matrix, always_set
        )
        if obj_info is None:
            continue

        layer_queue = area_queue.setdefault(area, {})  # if area has no queue, create it
        for mat_info in obj_info.mats:
            mat = mat_info[2]
            obj_queue = layer_queue.setdefault(int(mat.layer or "0"), {})  # if layer has no queue, create it
            if obj_name not in obj_queue:  # if obj not already present in the layer's obj queue, create a shallow copy
                obj_info = obj_queue[obj_name] = copy.copy(obj_info)
                obj_info.mats = []
            obj_queue[obj_name].mats.append(mat_info)

    for area, layer_queue in area_queue.items():
        render_state = area.render_state.copy()
        for layer, obj_queue in sorted(layer_queue.items(), key=lambda item: item[0]):  # sort by layer
            render_state.set_values_from_cache(layer_rendermodes[layer])
            for info in dict(sorted(obj_queue.items(), key=lambda item: item[0])):  # sort by obj name
                draw_f64_obj(render_engine, render_state, obj_queue[info])
