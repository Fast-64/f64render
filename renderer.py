import math
import struct
from typing import NamedTuple
import bpy
import mathutils
import gpu
from .utils.addon import addon_set_fast64_path
from .mesh.gpu_batch import create_vert_buf, batch_for_shader
from .material.parser import F64Material, f64_material_parse, node_material_parse
import pathlib
import time
import numpy as np

from .mesh.mesh import MeshBuffers, mesh_to_buffers

f64render_materials_dirty = True
f64render_instance = None
f64render_meshCache: dict[MeshBuffers] = {}
current_ucode = None

# N64 is y-up, blender is z-up
yup_to_zup = mathutils.Quaternion((1, 0, 0), math.radians(90.0)).to_matrix().to_4x4()

MISSING_TEXTURE_COLOR = (0, 0, 0, 1)

UNIFORM_BUFFER_STRUCT = struct.Struct(
  "8i"              # blender
  "16f"             # tile settings (mask/shift/low/high)
  "16i"             # color-combiner settings
  "i i i i"         # geoMode, other-low, other-high, flags
  "4f 4f 3f f 3f f" # light (first light direction W is alpha-clip)
  "4f 4f 4f"        # prim, env, ambient
  "2f 2f"           # prim_lod, prim-depth
  "8f 6f"           # ck center/scale, k0-k5,
)

def cache_del_by_mesh(mesh_name):
  global f64render_meshCache
  for key in list(f64render_meshCache.keys()):
    if f64render_meshCache[key].mesh_name == mesh_name:
      del f64render_meshCache[key]

def obj_has_f3d_materials(obj):
  for slot in obj.material_slots:
    if slot.material.is_f3d and slot.material.f3d_mat:
      return True
  return False

class Fast64RenderEngine(bpy.types.RenderEngine):
  bl_idname = "FAST64_RENDER_ENGINE"
  bl_label = "Fast64 Renderer"
  bl_use_preview = False

  def __init__(self):
    super().__init__()
    addon_set_fast64_path()

    self.shader = None
    self.shader_2d = None
    self.shader_fallback = None
    self.vbo_format = None
    self.draw_handler = None
    self.last_ucode = None

    self.time_count = 0
    self.time_total = 0
        
    self.depth_texture: gpu.types.GPUTexture = None
    self.color_texture: gpu.types.GPUTexture = None
    self.update_render_size(128, 128)
    bpy.app.handlers.depsgraph_update_post.append(Fast64RenderEngine.mesh_change_listener)

    if "f64render_missing_texture" not in bpy.data.images:
      # Create a 1x1 image
      bpy.data.images.new("f64render_missing_texture", 1, 1).pixels = MISSING_TEXTURE_COLOR

    ext_list = gpu.capabilities.extensions_get()
    self.shader_interlock_support = 'GL_ARB_fragment_shader_interlock' in ext_list
    if not self.shader_interlock_support:
      print("\n\nWarning: GL_ARB_fragment_shader_interlock not supported!\n\n")
    self.shader_info_img_impl = bpy.app.version >= (4, 1, 0)
    if not self.shader_info_img_impl:
      print("\n\nWarning: Blender version too old! Expect limited blending emulation!\n\n")
    self.draw_range_impl = bpy.app.version >= (3, 6, 0)

  def __del__(self):
    if Fast64RenderEngine.mesh_change_listener in bpy.app.handlers.depsgraph_update_post:
      bpy.app.handlers.depsgraph_update_post.remove(Fast64RenderEngine.mesh_change_listener)
    pass
  
  def update_render_size(self, size_x, size_y):
    if not self.depth_texture or size_x != self.depth_texture.width or size_y != self.depth_texture.height:
      self.depth_texture = gpu.types.GPUTexture((size_x, size_y), format='R32I')
      self.color_texture = gpu.types.GPUTexture((size_x, size_y), format='R32UI')

  def init_shader(self):
    if not self.shader:
      print("Compiling shader")

      shaderPath = (pathlib.Path(__file__).parent / "shader").resolve()
      shaderVert = ""
      shaderFrag = ""

      with open(shaderPath / "utils.glsl", "r", encoding="utf-8") as f:
        shaderUtils = f.read()
        shaderVert += shaderUtils
        shaderFrag += shaderUtils

      with open(shaderPath / "defines.glsl", "r", encoding="utf-8") as f:
        shaderDef = f.read()
        shaderVert += shaderDef
        shaderFrag += shaderDef

      with open(shaderPath / "main3d.vert.glsl", "r", encoding="utf-8") as f:
        shaderVert += f.read()

      with open(shaderPath / "main3d.frag.glsl", "r", encoding="utf-8") as f:
        shaderFrag += f.read()

      shader_info = gpu.types.GPUShaderCreateInfo()
      
      with open(shaderPath / "structs.glsl", "r", encoding="utf-8") as f:
        shader_info.typedef_source(f.read())
      
      # vertex -> fragment
      vert_out = gpu.types.GPUStageInterfaceInfo("vert_interface")
      vert_out.no_perspective("VEC4", "cc_shade")
      vert_out.flat("VEC4", "cc_shade_flat")
      vert_out.smooth("VEC4", "uv")
      vert_out.no_perspective("VEC2", "posScreen")
      vert_out.flat("VEC4", "tileSize")

      if self.shader_info_img_impl:
        shader_info.define("depth_unchanged", "depth_any")
        if self.shader_interlock_support:
          shader_info.define("USE_SHADER_INTERLOCK", "1")
        shader_info.define("BLEND_EMULATION", "1")

      shader_info.push_constant("MAT4", "matMVP")
      shader_info.push_constant("MAT3", "matNorm")

      shader_info.uniform_buf(0, "UBO_Material", "material")

      shader_info.vertex_in(0, "VEC3", "pos") # keep blenders name keep for better compat.
      shader_info.vertex_in(1, "VEC3", "inNormal")
      shader_info.vertex_in(2, "VEC4", "inColor")
      shader_info.vertex_in(3, "VEC2", "inUV")
      shader_info.vertex_out(vert_out)
      
      shader_info.sampler(0, "FLOAT_2D", "tex0")
      shader_info.sampler(1, "FLOAT_2D", "tex1")
      
      if self.shader_info_img_impl:
        shader_info.image(2, 'R32UI', "UINT_2D_ATOMIC", "color_texture", qualifiers={"READ", "WRITE"})
        shader_info.image(3, 'R32I',  "INT_2D_ATOMIC",  "depth_texture", qualifiers={"READ", "WRITE"})
      else:
        shader_info.fragment_out(0, "VEC4", "FragColor")

      shader_info.vertex_source(shaderVert)
      shader_info.fragment_source(shaderFrag)
      
      self.shader = gpu.shader.create_from_info(shader_info)      
      self.shader_fallback = gpu.shader.from_builtin('3D_UNIFORM_COLOR' if bpy.app.version < (4, 1, 0) else 'UNIFORM_COLOR')
      self.vbo_format = self.shader.format_calc()

  def init_shader_2d(self):
    if not self.shader_2d:
      print("Compiling 2D shader")
      # 2D shader (offscreen to viewport)
      shader_info = gpu.types.GPUShaderCreateInfo()
      vert_out = gpu.types.GPUStageInterfaceInfo("vert_2d")
      vert_out.smooth("VEC2", "uv")

      # Hacky workaround for blender forcing an early depth test ('layout(depth_unchanged) out float gl_FragDepth;')
      shader_info.define("depth_unchanged", "depth_any")
      shader_info.image(2, 'R32UI', "UINT_2D_ATOMIC", "color_texture", qualifiers={"READ"})

      shader_info.fragment_out(0, "VEC4", "FragColor")
      shader_info.vertex_in(0, "VEC2", "pos")
      shader_info.vertex_out(vert_out)

      shader_info.vertex_source("""
        void main() {
          gl_Position = vec4(pos, 0.0, 1.0);
          uv = pos.xy * 0.5 + 0.5;
        }""")
      
      shader_info.fragment_source("""
        void main() {
          ivec2 textureSize2d = imageSize(color_texture);
          ivec2 coord = ivec2(uv.xy * vec2(textureSize2d)); 
          FragColor =  unpackUnorm4x8(imageLoad(color_texture, coord).r);
          gl_FragDepth = 0.99999;
        }""")
      
      self.shader_2d = gpu.shader.create_from_info(shader_info)                             

  def mesh_change_listener(scene, depsgraph):
    global f64render_meshCache
    global f64render_materials_dirty
    global current_ucode
    # print("################ MESH CHANGE LISTENER ################")  

    if depsgraph.id_type_updated('SCENE'):
      if current_ucode != depsgraph.scene.f3d_type:
        f64render_materials_dirty = True
        current_ucode = depsgraph.scene.f3d_type

    if depsgraph.id_type_updated('MATERIAL'):
      for update in depsgraph.updates:
        # this seems to trigger for all materials if only one changed (@TODO: check if i can get proper updates)
        f64render_materials_dirty = True

    if depsgraph.id_type_updated('OBJECT'):
      for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object) and update.id.type in {"MESH", "CURVE", "SURFACE", "FONT"}:
          cache_del_by_mesh(update.id.data.name)

  def view_update(self, context, depsgraph):
    if self.draw_handler is None:
      self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_scene, (context, depsgraph), 'WINDOW', 'POST_VIEW')

    # this causes the mesh to update during edit-mode
    for obj in depsgraph.objects:
      if obj.type == 'MESH' and obj.mode == 'EDIT':
        meshID = obj.name + "#" + obj.data.name
        if meshID in f64render_meshCache:
          del f64render_meshCache[meshID]

  def view_draw(self, context, depsgraph):
    self.draw_scene(context, depsgraph)

  def draw_mat(self, render_obj: MeshBuffers, mat_idx: int, indices_count: int):
    if self.draw_range_impl:
      render_obj.batch.draw_range(self.shader, elem_start=render_obj.index_offsets[mat_idx] * 3, elem_count=indices_count)
    else:
      render_obj.batch[mat_idx].draw(self.shader)

  def draw_scene(self, context, depsgraph):
    global f64render_meshCache
    global f64render_materials_dirty
    
    # TODO: fixme, after reloading this script during dev, something calls this function
    #       with an invalid reference (viewport?)
    if repr(self).endswith("invalid>"):
        return

    t = time.process_time()

    space_view_3d = context.space_data
    if self.shader_info_img_impl:
      self.update_render_size(context.region.width, context.region.height)
      self.color_texture.clear(format='UINT', value=[0x080808])
      self.depth_texture.clear(format='INT', value=[0])

    self.init_shader()
    if self.shader_info_img_impl:
      self.init_shader_2d()
    self.shader.bind()

    # Enable depth test
    gpu.state.depth_test_set('LESS')
    gpu.state.depth_mask_set(True)

    # global params
    fast64_rs = depsgraph.scene.fast64.renderSettings
    f64render_rs: F64RenderSettings = depsgraph.scene.f64render.render_settings
    lightDir0, lightDir1 = fast64_rs.light0Direction, fast64_rs.light1Direction
    if not fast64_rs.useWorldSpaceLighting:
      view_rotation = (mathutils.Quaternion((1, 0, 0), math.radians(90.0)) @ context.region_data.view_matrix.to_quaternion()).to_matrix()
      lightDir0, lightDir1 = lightDir0 @ view_rotation, lightDir1 @ view_rotation

    # Note: space conversion to Y-up happens indirectly during the normal matrix calculation
    lightColor0 = fast64_rs.light0Color
    lightColor1 = fast64_rs.light1Color
    ambientColor = fast64_rs.ambientColor

    lastPrimColor = f64render_rs.default_prim_color
    last_prim_lod = np.array([0, 0], dtype=np.float32)
    lastEnvColor = f64render_rs.default_env_color
    last_ck = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    last_convert = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    fallback_objs = []
    # get visible objects, this cannot be done in despgraph objects for whatever reason
    hidden_obj = [ob.name for ob in bpy.context.view_layer.objects if not ob.visible_get() and ob.data is not None]

    class TempObjRenderInfo(NamedTuple):
      mvp_matrix: mathutils.Matrix
      normal_matrix: mathutils.Matrix
      render_obj: MeshBuffers
      mats: list[tuple[int, F64Material]]

    render_queue: dict[int, dict[bpy.types.Object, TempObjRenderInfo]] = {}
    for obj in depsgraph.objects:
      if obj.type in {"MESH", "CURVE", "SURFACE", "FONT"} and obj.data is not None:

        meshID = obj.name + "#" + obj.data.name

        # check for objects that transitioned from non-f3d to f3d materials
        if meshID in f64render_meshCache:
          renderObj = f64render_meshCache[meshID]
          if len(renderObj.mat_data) == 0 and obj_has_f3d_materials(obj):
            del f64render_meshCache[meshID]
            
        # Mesh not cached: parse & convert mesh data, then prepare a GPU batch
        if meshID not in f64render_meshCache:
          # print("    -> Update object", meshID)
          if obj.mode == 'EDIT':
            mesh = obj.evaluated_get(depsgraph).to_mesh()
          else:
            mesh = obj.evaluated_get(depsgraph).to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

          renderObj = f64render_meshCache[meshID] = mesh_to_buffers(mesh)
          renderObj.mesh_name = obj.data.name

          mat_count = len(obj.material_slots)
          vert_buf = create_vert_buf(self.vbo_format,
            renderObj.vert,
            renderObj.norm,
            renderObj.color,
            renderObj.uv,
          )
          if self.draw_range_impl:
            renderObj.batch = batch_for_shader(vert_buf, renderObj.indices)
          else: # we need to create batches for each material
            renderObj.batch = []
            if not obj.material_slots: # if no material slot, we only have one batch for the whole geo
              renderObj.batch = [batch_for_shader(vert_buf, renderObj.indices)]
            else:
              renderObj.batch = []      
            for i, slot in enumerate(obj.material_slots):
              indices = renderObj.indices[renderObj.index_offsets[i]:renderObj.index_offsets[i+1]]
              if len(indices) == 0: # ignore unused materials
                renderObj.batch.append(None)
              else:
                renderObj.batch.append(batch_for_shader(vert_buf, indices))

          ubo_size = UNIFORM_BUFFER_STRUCT.size
          ubo_size = (ubo_size + 15) & ~15 # force 16-byte alignment

          renderObj.mat_data = [bytes(ubo_size)] * mat_count

          renderObj.ubo_mat_data = [None] * mat_count
          renderObj.materials = [None] * mat_count

          for i in range(mat_count):
            renderObj.ubo_mat_data[i] = gpu.types.GPUUniformBuf(renderObj.mat_data[i])

          obj.to_mesh_clear()
        
        if not obj_has_f3d_materials(obj):
          fallback_objs.append(obj)
          continue

        if obj.data is None: continue
        # Handle "Local View" (pressing '/')
        if space_view_3d.local_view and not obj.local_view_get(space_view_3d): continue
        if obj.name in hidden_obj: continue
        # print("Draw object", obj.data.session_uid, visible_obj_ids)
  
        # print("space_view_3d.local_view", space_view_3d.clip_start, space_view_3d.clip_end)

        meshID = obj.name + "#" + obj.data.name
        if meshID not in f64render_meshCache: continue
        # print("  -> Draw object", meshID)
        render_obj: MeshBuffers = f64render_meshCache[meshID]

        modelview_matrix = obj.matrix_world
        projection_matrix = context.region_data.perspective_matrix
        mvp_matrix = projection_matrix @ modelview_matrix
        normal_matrix = (context.region_data.view_matrix @ obj.matrix_world).to_3x3().inverted().transposed()
        for i, slot in enumerate(obj.material_slots):
          indices_count = (render_obj.index_offsets[i+1] - render_obj.index_offsets[i]) * 3
          if indices_count == 0: # ignore unused materials
            continue
          
          f3d_mat = slot.material.f3d_mat                    
          if f64render_materials_dirty or render_obj.materials[i] is None:
            render_obj.materials[i] = f64_material_parse(f3d_mat, render_obj.materials[i])

          f64mat = render_obj.materials[i]
          obj_queue = render_queue.setdefault(f64mat.layer, {})
          info = obj_queue.setdefault(
            obj, TempObjRenderInfo(mvp_matrix, normal_matrix, render_obj, [])
          )
          info.mats.append((i, indices_count, f64mat))
    for i, layer in render_queue.items():
      render_queue[i] = dict(sorted(layer.items(), key=lambda item: item[0].name))
    render_queue = dict(sorted(render_queue.items(), key=lambda item: item[0]))

    if self.shader_info_img_impl:
      self.shader.image('depth_texture', self.depth_texture)
      self.shader.image('color_texture', self.color_texture)

    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)
    gpu.state.blend_set("NONE")

    # Draw materials in layer order
    for obj_queue in render_queue.values():
      for obj, info in obj_queue.items():
        self.shader.uniform_float("matMVP", info.mvp_matrix)
        self.shader.uniform_float("matNorm", info.normal_matrix)
   
        for mat_idx, indices_count, f64mat in info.mats:
          gpu.state.face_culling_set(f64mat.cull)
          if not self.shader_info_img_impl:
            gpu.state.blend_set(f64mat.blend)
            gpu.state.depth_test_set(f64mat.depth_test)
            gpu.state.depth_mask_set(f64mat.depth_write)

          if f64mat.tex0Buff: self.shader.uniform_sampler("tex0", f64mat.tex0Buff)
          if f64mat.tex1Buff: self.shader.uniform_sampler("tex1", f64mat.tex1Buff)

          info.render_obj.mat_data[mat_idx] = UNIFORM_BUFFER_STRUCT.pack(
            *f64mat.blender,
            *f64mat.tile_conf,
            *f64mat.cc,
            f64mat.geo_mode,
            f64mat.othermode_l,
            f64mat.othermode_h,
            f64mat.flags,
            *(f64mat.color_light if f64mat.set_light      else lightColor0),
            *lightColor1,
            *lightDir0,
            f64mat.alphaClip,
            *lightDir1,
            0,
            *(f64mat.color_prim    if f64mat.set_prim     else lastPrimColor),
            *(f64mat.color_env     if f64mat.set_env      else lastEnvColor),
            *(f64mat.color_ambient if f64mat.set_ambient  else ambientColor),
            *(f64mat.ck            if f64mat.set_ck       else last_ck),
            *(f64mat.lod_prim      if f64mat.set_prim     else last_prim_lod),
            *f64mat.prim_depth,
            *(f64mat.convert       if f64mat.set_convert  else last_convert),
          )

          if f64mat.set_prim: lastPrimColor, last_prim_lod = f64mat.color_prim, f64mat.lod_prim
          if f64mat.set_env: lastEnvColor = f64mat.color_env
          if f64mat.set_ck: last_ck = f64mat.ck
          if f64mat.set_convert: last_convert = f64mat.convert
          
          info.render_obj.ubo_mat_data[mat_idx].update(info.render_obj.mat_data[mat_idx])                        
          self.shader.uniform_block("material", info.render_obj.ubo_mat_data[mat_idx])
          
          # @TODO: frustum-culling (blender doesn't do it)
          self.draw_mat(info.render_obj, mat_idx, indices_count)

    f64render_materials_dirty = False
    draw_time = (time.process_time() - t) * 1000
    self.time_total += draw_time
    self.time_count += 1
    #print("Time F3D (ms)", draw_time)

    if self.time_count > 20:
      print("Time F3D AVG (ms)", self.time_total / self.time_count, self.time_count)
      self.time_total = 0
      self.time_count = 0
          
    if len(fallback_objs) > 0:
      t = time.process_time()
      self.shader_fallback.bind()

      for obj in fallback_objs:
        meshID = obj.name + "#" + obj.data.name
        renderObj = f64render_meshCache[meshID]

        if obj.name in hidden_obj: continue

        # get material (we don't expect any changes here, so caching is fine)
        if renderObj.materials is None or len(renderObj.materials) == 0:
          renderObj.materials = [F64Material()]
          if obj.material_slots:
            mat = obj.material_slots[0].materials[0]
            renderObj.materials[0] = node_material_parse(mat)

        self.shader_fallback.uniform_float("color", renderObj.materials[0].color_prim)

        modelview_matrix = obj.matrix_world
        projection_matrix = context.region_data.perspective_matrix
        mvp_matrix = projection_matrix @ modelview_matrix
        self.shader_fallback.uniform_float("ModelViewProjectionMatrix", mvp_matrix)

        self.draw_mat(renderObj, 0, len(renderObj.indices))
        obj.to_mesh_clear()

      #print("Time fallback (ms)", (time.process_time() - t) * 1000)

    if not self.shader_info_img_impl:
      return # when there's no access to color and depth aux images, we render directly, so skip final 2d draw

    #t = time.process_time()
    gpu.state.face_culling_set('NONE')
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set('LESS')
    gpu.state.depth_mask_set(False)

    self.shader_2d.bind()
    
    # @TODO: why can't i cache this?
    vbo_2d = gpu.types.GPUVertBuf(self.shader_2d.format_calc(), 6)
    vbo_2d.attr_fill("pos", [(-1, -1), (-1, 1), (1, 1), (1, 1), (1, -1), (-1, -1)])
    batch_2d = gpu.types.GPUBatch(type="TRIS", buf=vbo_2d)

    self.shader_2d.image('color_texture', self.color_texture)
    batch_2d.draw(self.shader_2d)

    #print("Time 2D (ms)", (time.process_time() - t) * 1000)

class F64RenderSettings(bpy.types.PropertyGroup):
  default_prim_color: bpy.props.FloatVectorProperty(
    name="Default Prim Color",
    default=(1, 1, 1, 1),
    subtype="COLOR",
    size=4,
    min=0,
    max=1,
  )
  default_env_color: bpy.props.FloatVectorProperty(
    name="Default Env Color",
    default=(0.5, 0.5, 0.5, 0.5),
    subtype="COLOR",
    size=4,
    min=0,
    max=1,
  )

class F64RenderProperties(bpy.types.PropertyGroup):
  render_settings: bpy.props.PointerProperty(type=F64RenderSettings)

class F64RenderSettingsPanel(bpy.types.Panel):
  bl_label = "f64render"
  bl_idname = "OBJECT_PT_F64RENDER_SETTINGS_PANEL"
  bl_space_type = "VIEW_3D"
  bl_region_type = "WINDOW"

  def draw(self, context):
    layout = self.layout
    f64render_rs: F64RenderSettings = context.scene.f64render.render_settings
    layout.prop(f64render_rs, "default_prim_color")
    layout.prop(f64render_rs, "default_env_color")

def draw_render_settings(self, context):
  if context.scene.render.engine == Fast64RenderEngine.bl_idname:
    self.layout.popover(F64RenderSettingsPanel.bl_idname)

# By default blender will hide quite a few panels like materials or vertex attributes
# Add this method to override the check blender does by render engine
def get_panels():
    exclude_panels = {
      'VIEWLAYER_PT_filter',
        'VIEWLAYER_PT_layer_passes',
    }
    
    include_panels = {
      'EEVEE_MATERIAL_PT_context_material',
      'MATERIAL_PT_preview'
    }

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
      if hasattr(panel, 'COMPAT_ENGINES'):
        if (('BLENDER_RENDER' in panel.COMPAT_ENGINES and panel.__name__ not in exclude_panels)
          or panel.__name__ in include_panels):
          panels.append(panel)

    return panels

def register():
  global f64render_meshCache
  f64render_meshCache = {}

  bpy.types.RenderEngine.f64_render_engine = bpy.props.PointerProperty(type=Fast64RenderEngine)
  for panel in get_panels():
    panel.COMPAT_ENGINES.add('FAST64_RENDER_ENGINE')

  bpy.types.Scene.f64render = bpy.props.PointerProperty(type=F64RenderProperties)

  bpy.types.VIEW3D_HT_header.append(draw_render_settings)

def unregister():
  global f64render_meshCache
  f64render_meshCache = {}

  bpy.types.VIEW3D_HT_header.remove(draw_render_settings)

  del bpy.types.RenderEngine.f64_render_engine

  for panel in get_panels():
    if 'FAST64_RENDER_ENGINE' in panel.COMPAT_ENGINES:
      panel.COMPAT_ENGINES.remove('FAST64_RENDER_ENGINE')

  del bpy.types.Scene.f64render
