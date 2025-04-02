import bpy

class F64Globals:
  def __init__(self):
    self.clear()
  def clear(self):
    self.materials_cache: dict[bpy.types.Material, "F64Material"] = {}
    self.meshCache: dict["MeshBuffers"] = {}
    self.obj_lights: dict[str, "F64Light"] = {}
    self.area_lookup: dict|None = None
    self.current_ucode = None

F64_GLOBALS = F64Globals()