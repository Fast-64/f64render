import copy
from typing import NamedTuple

import bpy
import mathutils

from .material.parser import parse_f3d_rendermode_preset, F64RenderState
from .common import ObjRenderInfo, draw_f64_obj, collect_obj_info, get_scene_render_state
from .properties import F64RenderSettings
from .globals import F64_GLOBALS


class RoomRenderInfo(NamedTuple):
    render_state: F64RenderState
    name: str

    def __hash__(self):
        return hash(self.name)


def get_oot_room_childrens(scene: bpy.types.Scene):
    if F64_GLOBALS.oot_room_lookup is not None:
        return F64_GLOBALS.oot_room_lookup

    oot_room_lookup = {}
    render_state = get_scene_render_state(scene)
    scene_objs: list[bpy.types.Object] = []
    room_objs: list[bpy.types.Object] = []

    def get_room_children(obj: bpy.types.Object, name: str):
        for child in sorted(obj.children, key=lambda item: item.name):
            if child not in room_objs:
                oot_room_lookup[child.name] = RoomRenderInfo(render_state, name)
                get_room_children(child, name)
            else:
                get_room_children(child, child.name)

    def get_scene_children(obj: bpy.types.Object, name: str):
        for child in sorted(obj.children, key=lambda item: item.name):
            if child in room_objs:
                get_room_children(child, child.name)
            else:
                oot_room_lookup[child.name] = RoomRenderInfo(render_state, name)
                get_scene_children(child, name)

    for obj in bpy.data.objects:
        if obj.type == "EMPTY":
            if obj.ootEmptyType == "Scene":
                scene_objs.append(obj)
            if obj.ootEmptyType == "Room":
                room_objs.append(obj)

    for scene_obj in scene_objs:
        get_scene_children(scene_obj, scene_obj.name)

    fake_room = RoomRenderInfo(render_state, "")
    for obj in bpy.data.objects:
        if obj.name not in oot_room_lookup:
            oot_room_lookup[obj.name] = fake_room

    F64_GLOBALS.oot_room_lookup = oot_room_lookup
    return oot_room_lookup


# TODO if porting to fast64, reuse existing default layer dict
DEFAULT_LAYERS = {
    "Opaque": ("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"),
    "Transparent": ("G_RM_AA_ZB_XLU_SURF", "G_RM_AA_ZB_XLU_SURF2"),
    "Overlay": ("G_RM_AA_ZB_OPA_SURF", "G_RM_AA_ZB_OPA_SURF2"),
}


def draw_oot_scene(
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
    for layer, (cycle1, cycle2) in DEFAULT_LAYERS.items():
        if world:
            defaults = world.ootDefaultRenderModes
            cycle1, cycle2 = (getattr(defaults, f"{layer.lower()}Cycle{cycle}") for cycle in (1, 2))
        rm_state = F64RenderState()
        rm_state.set_from_rendermode(parse_f3d_rendermode_preset(cycle1, cycle2))
        rm_state.save_cache()
        layer_rendermodes[layer] = rm_state

    ignore, collision = f64render_rs.render_type == "IGNORE", f64render_rs.render_type == "COLLISION"
    specific_room = f64render_rs.oot_specific_room.name if f64render_rs.oot_specific_room else None
    room_lookup = get_oot_room_childrens(depsgraph.scene)
    layer_queue: dict[str, dict[RoomRenderInfo, dict[str, ObjRenderInfo]]] = {}

    for obj in depsgraph.objects:
        obj_name = obj.name
        room = room_lookup[obj_name]
        if (
            (ignore and obj.ignore_render)
            or (collision and obj.ignore_collision)
            or (specific_room and room.name != specific_room)
        ):
            continue
        obj_info = collect_obj_info(
            render_engine, obj, depsgraph, hidden_objs_names, space_view_3d, projection_matrix, view_matrix, always_set
        )
        if obj_info is None:
            continue

        for mat_info in obj_info.mats:
            mat = mat_info[2]
            room_queue = layer_queue.setdefault(mat.layer or "Opaque", {})  # if layer has no room queue, create it
            obj_queue = room_queue.setdefault(room, {})  # if current room has no obj queue in this layer, create it
            if obj_name not in obj_queue:  # if obj not already present in the layer's obj queue, create a shallow copy
                obj_info = obj_queue[obj_name] = copy.copy(obj_info)
                obj_info.mats = []
            obj_queue[obj_name].mats.append(mat_info)

    for layer in ("Opaque", "Transparent", "Overlay"):
        room_queue = layer_queue.get(layer)
        if room_queue is None:
            continue
        # sort by room name, this doesn't correspond to something the fast64 exporter or the game rendering does
        # but it at least helps make the behavior reproducible
        for room, obj_queue in sorted(room_queue.items(), key=lambda item: item[0].name):
            render_state = room.render_state.copy()
            render_state.set_values_from_cache(layer_rendermodes.get(layer, layer_rendermodes["Opaque"]))
            for info in dict(sorted(obj_queue.items(), key=lambda item: item[0])):  # sort by obj name
                draw_f64_obj(render_engine, render_state, obj_queue[info])
