import copy
from typing import NamedTuple

import bpy
import mathutils

from .material.parser import parse_f3d_rendermode_preset, F64RenderState
from .common import (
    SCENE_UNIFORM_BUFFER_STRUCT,
    ObjRenderInfo,
    draw_f64_obj,
    collect_obj_info,
    get_scene_render_state,
    update_scene_ubo_generic,
)
from .properties import F64RenderSettings
from .globals import F64_GLOBALS


def get_time_of_day_settings(scene: bpy.types.Object):
    time_of_day_lights = scene.ootSceneHeader.timeOfDayLights
    return getattr(time_of_day_lights, time_of_day_lights.menuTab.lower())


class RoomRenderInfo(NamedTuple):
    render_state: F64RenderState
    name: str

    def __hash__(self):
        return hash(self.name)


class SceneRenderInfo(NamedTuple):
    render_state: F64RenderState
    name: str
    obj: bpy.types.Object


def get_oot_room_childrens(scene: bpy.types.Scene):
    if F64_GLOBALS.oot_room_lookup is not None and F64_GLOBALS.oot_scene_lookup is not None:
        return F64_GLOBALS.oot_room_lookup, F64_GLOBALS.oot_scene_lookup

    oot_room_lookup, oot_scene_lookup = {}, {}
    render_state = get_scene_render_state(scene)
    scene_objs: list[bpy.types.Object] = []
    room_objs: list[bpy.types.Object] = []

    def get_room_info(obj: bpy.types.Object):
        return RoomRenderInfo(render_state, obj.name)

    def get_room_children(obj: bpy.types.Object, render_info: RoomRenderInfo, scene_info: SceneRenderInfo = None):
        for child in sorted(obj.children, key=lambda item: item.name):
            if scene_info is not None:
                oot_scene_lookup[child.name] = scene_info
            if child not in room_objs:
                oot_room_lookup[child.name] = render_info
                get_room_children(child, render_info, scene_info)
            else:
                get_room_children(child, get_room_info(child), scene_info)

    def get_scene_children(obj: bpy.types.Object, scene_info: SceneRenderInfo):
        for child in sorted(obj.children, key=lambda item: item.name):
            oot_scene_lookup[child.name] = scene_info
            if child in room_objs:
                get_room_children(child, get_room_info(child), scene_info)

    for obj in bpy.data.objects:
        if obj.type == "EMPTY":
            if obj.ootEmptyType == "Scene":
                scene_objs.append(obj)
            if obj.ootEmptyType == "Room":
                room_objs.append(obj)

    for scene_obj in scene_objs:
        render_state_copy = render_state.copy()
        time_of_day = get_time_of_day_settings(scene_obj)
        render_state_copy.fog.pos = (time_of_day.fogNear, 1000)
        render_state.save_cache()
        oot_scene_lookup[scene_obj.name] = SceneRenderInfo(render_state, scene_obj.name, scene_obj)
        get_scene_children(scene_obj, oot_scene_lookup[scene_obj.name])

    fake_room = oot_room_lookup[""] = RoomRenderInfo(render_state, "")
    fake_scene = oot_scene_lookup[""] = SceneRenderInfo(render_state, "", None)
    for obj in bpy.data.objects:
        if obj.name not in oot_room_lookup:
            oot_room_lookup[obj.name] = fake_room
        if obj.name not in oot_scene_lookup:
            oot_scene_lookup[obj.name] = fake_scene

    F64_GLOBALS.oot_room_lookup = oot_room_lookup
    F64_GLOBALS.oot_scene_lookup = oot_scene_lookup
    return oot_room_lookup, oot_scene_lookup


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
    from fast64_internal.utility import get_blender_to_game_scale

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
    room_lookup, scene_lookup = get_oot_room_childrens(depsgraph.scene)
    layer_queue: dict[str, dict[RoomRenderInfo, dict[SceneRenderInfo, dict[str, ObjRenderInfo]]]] = {}

    for obj in depsgraph.objects:
        obj_name = obj.name
        room = room_lookup[obj_name]
        scene = scene_lookup[obj_name]
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
            scene_queue = layer_queue.setdefault(mat.layer or "Opaque", {})  # if layer has no room queue, create it
            room_queue = scene_queue.setdefault(scene.name, {})
            obj_queue = room_queue.setdefault(
                room.name, {}
            )  # if current room has no obj queue in this layer, create it
            if obj_name not in obj_queue:  # if obj not already present in the layer's obj queue, create a shallow copy
                obj_info = obj_queue[obj_name] = copy.copy(obj_info)
                obj_info.mats = []
            obj_queue[obj_name].mats.append(mat_info)

    for layer in ("Opaque", "Transparent", "Overlay"):
        scene_queue = layer_queue.get(layer)
        if scene_queue is None:
            continue
        for scene_name in sorted(scene_queue.keys(), key=lambda item: item):
            scene = scene_lookup[scene_name]
            render_state = scene.render_state.copy()
            if scene_name == "":
                update_scene_ubo_generic(render_engine, depsgraph.scene)
            else:
                current = get_time_of_day_settings(scene.obj)
                render_engine.scene_ubo.update(
                    SCENE_UNIFORM_BUFFER_STRUCT.pack(
                        *(round(x) for x in (10, current.z_far * 1000)), get_blender_to_game_scale(bpy.context)
                    )
                )
            room_queue = scene_queue.get(scene_name)
            if room_queue is None:
                continue
            # sort by room name, this doesn't correspond to something the fast64 exporter or the game rendering does
            # but it at least helps make the behavior reproducible
            for room_name, obj_queue in sorted(room_queue.items(), key=lambda item: item[0]):
                room = room_lookup[room_name]
                render_state.set_values_from_cache(room.render_state)
                render_state.set_values_from_cache(layer_rendermodes.get(layer, layer_rendermodes["Opaque"]))
                for info in dict(sorted(obj_queue.items(), key=lambda item: item[0])):  # sort by obj name
                    draw_f64_obj(render_engine, render_state, obj_queue[info])
