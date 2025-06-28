import bpy


class F64Globals:
    def __init__(self):
        self.clear()

    def clear(self):
        self.materials_cache: dict[bpy.types.Material, "F64Material"] = {}
        self.meshCache: dict["MeshBuffers"] = {}
        self.obj_lights: dict[str, "F64Light"] = {}
        self.sm64_area_lookup: dict | None = None
        self.oot_room_lookup: dict | None = None  # oot
        self.rebuid_shaders = True
        self.current_ucode = self.world_lighting = self.current_gamemode = None

    def clear_areas(self):
        self.sm64_area_lookup = None
        self.oot_room_lookup = None


F64_GLOBALS = F64Globals()
