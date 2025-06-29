import math
import pathlib
import time

import bpy
import mathutils
import gpu

from .utils.addon import addon_set_fast64_path
from .material.parser import f64_parse_obj_light
from .common import ObjRenderInfo, draw_f64_obj, get_scene_render_state, collect_obj_info
from .properties import F64RenderProperties, F64RenderSettings
from .globals import F64_GLOBALS

from .sm64 import draw_sm64_scene
from .oot import draw_oot_scene

# N64 is y-up, blender is z-up
yup_to_zup = mathutils.Quaternion((1, 0, 0), math.radians(90.0)).to_matrix().to_4x4()

MISSING_TEXTURE_COLOR = (0, 0, 0, 1)


def cache_del_by_mesh(mesh_name):
    for key in list(F64_GLOBALS.meshCache.keys()):
        if F64_GLOBALS.meshCache[key].mesh_name == mesh_name:
            del F64_GLOBALS.meshCache[key]


def obj_has_f3d_materials(obj):
    for slot in obj.material_slots:
        if slot.material.is_f3d and slot.material.f3d_mat:
            return True
    return False


def materials_set_light_direction(scene):
    return not (scene.gameEditorMode == "SM64" and scene.fast64.sm64.matstack_fix)


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
        self.use_atomic_rendering = True

        self.last_used_textures: dict[int, gpu.types.GPUTexture] = {}

        self.time_count = 0
        self.time_total = 0

        self.depth_texture: gpu.types.GPUTexture = None
        self.color_texture: gpu.types.GPUTexture = None
        self.update_render_size(128, 128)

        bpy.app.handlers.depsgraph_update_post.append(Fast64RenderEngine.mesh_change_listener)
        bpy.app.handlers.frame_change_post.append(Fast64RenderEngine.mesh_change_listener)
        bpy.app.handlers.load_pre.append(Fast64RenderEngine.on_file_load)

        if "f64render_missing_texture" not in bpy.data.images:
            # Create a 1x1 image
            bpy.data.images.new("f64render_missing_texture", 1, 1).pixels = MISSING_TEXTURE_COLOR

        ext_list = gpu.capabilities.extensions_get()
        self.shader_interlock_support = "GL_ARB_fragment_shader_interlock" in ext_list
        if not self.shader_interlock_support:
            print("\n\nWarning: GL_ARB_fragment_shader_interlock not supported!\n\n")
        if bpy.app.version < (4, 1, 0):
            print("\n\nWarning: Blender version too old! Expect limited blending emulation!\n\n")
        self.draw_range_impl = bpy.app.version >= (3, 6, 0)

    def __del__(self):
        def remove_handler(handler, func):
            while func in handler:
                handler.remove(func)

        remove_handler(bpy.app.handlers.depsgraph_update_post, Fast64RenderEngine.mesh_change_listener)
        remove_handler(bpy.app.handlers.frame_change_post, Fast64RenderEngine.mesh_change_listener)
        remove_handler(bpy.app.handlers.load_pre, Fast64RenderEngine.on_file_load)

    def update_render_size(self, size_x, size_y):
        if not self.depth_texture or size_x != self.depth_texture.width or size_y != self.depth_texture.height:
            self.depth_texture = gpu.types.GPUTexture((size_x, size_y), format="R32I")
            self.color_texture = gpu.types.GPUTexture((size_x, size_y), format="R32UI")

    def init_shader(self, scene: bpy.types.Scene):
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
        vert_out.smooth("VEC2", "inputUV")
        vert_out.no_perspective("VEC2", "posScreen")

        if self.use_atomic_rendering:
            shader_info.define("depth_unchanged", "depth_any")
            if self.shader_interlock_support:
                shader_info.define("USE_SHADER_INTERLOCK", "1")
            shader_info.define("BLEND_EMULATION", "1")
        # Using the already calculated view space normals instead of transforming the light direction makes
        # for cleaner and faster code
        shader_info.define("VIEWSPACE_LIGHTING", "0" if scene.fast64.renderSettings.useWorldSpaceLighting else "1")
        shader_info.define("SIMULATE_LOW_PRECISION", "1")

        shader_info.push_constant("MAT4", "matMVP")
        shader_info.push_constant("MAT3", "matNorm")

        shader_info.uniform_buf(0, "UBO_Material", "material")

        shader_info.vertex_in(0, "VEC3", "pos")  # keep blenders name keep for better compat.
        shader_info.vertex_in(1, "VEC3", "inNormal")
        shader_info.vertex_in(2, "VEC4", "inColor")
        shader_info.vertex_in(3, "VEC2", "inUV")
        shader_info.vertex_out(vert_out)

        for i in range(8):
            shader_info.sampler(i, "FLOAT_2D", f"tex{i}")

        if self.use_atomic_rendering:
            shader_info.image(2, "R32UI", "UINT_2D_ATOMIC", "color_texture", qualifiers={"READ", "WRITE"})
            shader_info.image(3, "R32I", "INT_2D_ATOMIC", "depth_texture", qualifiers={"READ", "WRITE"})
        else:
            shader_info.fragment_out(0, "VEC4", "FragColor")

        shader_info.vertex_source(shaderVert)
        shader_info.fragment_source(shaderFrag)

        self.shader = gpu.shader.create_from_info(shader_info)
        self.shader_fallback = gpu.shader.from_builtin(
            "3D_UNIFORM_COLOR" if bpy.app.version < (4, 1, 0) else "UNIFORM_COLOR"
        )
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
            shader_info.image(2, "R32UI", "UINT_2D_ATOMIC", "color_texture", qualifiers={"READ"})

            shader_info.fragment_out(0, "VEC4", "FragColor")
            shader_info.vertex_in(0, "VEC2", "pos")
            shader_info.vertex_out(vert_out)

            shader_info.vertex_source(
                """
        void main() {
          gl_Position = vec4(pos, 0.0, 1.0);
          uv = pos.xy * 0.5 + 0.5;
        }"""
            )

            shader_info.fragment_source(
                """
        void main() {
          ivec2 textureSize2d = imageSize(color_texture);
          ivec2 coord = ivec2(uv.xy * vec2(textureSize2d)); 
          FragColor =  unpackUnorm4x8(imageLoad(color_texture, coord).r);
          gl_FragDepth = 0.99999;
        }"""
            )

            self.shader_2d = gpu.shader.create_from_info(shader_info)

    def mesh_change_listener(scene, depsgraph):
        # print("################ MESH CHANGE LISTENER ################")

        for update in depsgraph.updates:
            if isinstance(update.id, bpy.types.Scene):
                if (
                    F64_GLOBALS.current_ucode != update.id.f3d_type
                    or F64_GLOBALS.current_gamemode != update.id.gameEditorMode
                ):
                    F64_GLOBALS.materials_cache = {}
                    F64_GLOBALS.current_ucode, F64_GLOBALS.current_gamemode = (
                        update.id.f3d_type,
                        update.id.gameEditorMode,
                    )
                world_lighting = update.id.fast64.renderSettings.useWorldSpaceLighting
                if world_lighting != F64_GLOBALS.world_lighting:
                    F64_GLOBALS.world_lighting = world_lighting
                    F64_GLOBALS.rebuid_shaders = True

                F64_GLOBALS.clear_areas()  # reset area lookup to refresh initial render state, is this the best approach?
            if isinstance(update.id, bpy.types.Material) and update.id in F64_GLOBALS.materials_cache:
                F64_GLOBALS.materials_cache.pop(update.id)
            is_obj_update = isinstance(update.id, bpy.types.Object)

            # support animating lights without uncaching materials, check if a light object was updated
            if (
                is_obj_update and isinstance(update.id.data, bpy.types.Light)
            ) and update.id.name in F64_GLOBALS.obj_lights:
                f64_parse_obj_light(
                    F64_GLOBALS.obj_lights[update.id.name], update.id, materials_set_light_direction(depsgraph.scene)
                )
            if is_obj_update and update.id.type in {"MESH", "CURVE", "SURFACE", "FONT"}:
                F64_GLOBALS.clear_areas()
                if update.is_updated_geometry:
                    cache_del_by_mesh(update.id.data.name)

    @bpy.app.handlers.persistent
    def on_file_load(_context):
        F64_GLOBALS.clear()

    def view_update(self, context, depsgraph):
        if self.draw_handler is None:
            self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_scene, (context, depsgraph), "WINDOW", "POST_VIEW"
            )

        # this causes the mesh to update during edit-mode
        for obj in depsgraph.objects:
            if obj.type == "MESH" and obj.mode == "EDIT":
                meshID = obj.name + "#" + obj.data.name
                if meshID in F64_GLOBALS.meshCache:
                    del F64_GLOBALS.meshCache[meshID]

    def view_draw(self, context, depsgraph):
        self.draw_scene(context, depsgraph)
        return  # uncomment to profile individual functions
        from cProfile import Profile
        from pstats import SortKey, Stats

        with Profile() as profile:
            self.draw_scene(context, depsgraph)
            Stats(profile).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()

    def draw_scene(self, context, depsgraph):
        # TODO: fixme, after reloading this script during dev, something calls this function
        #       with an invalid reference (viewport?)
        if repr(self).endswith("invalid>"):
            return

        t = time.process_time()

        space_view_3d = context.space_data
        f64render_rs: F64RenderSettings = depsgraph.scene.f64render.render_settings
        always_set = f64render_rs.always_set
        projection_matrix, view_matrix = context.region_data.perspective_matrix, context.region_data.view_matrix
        self.use_atomic_rendering = bpy.app.version >= (4, 1, 0) and f64render_rs.use_atomic_rendering

        if F64_GLOBALS.rebuid_shaders or self.shader is None:
            F64_GLOBALS.rebuid_shaders = False
            self.init_shader(context.scene)

        if self.use_atomic_rendering:
            self.update_render_size(context.region.width, context.region.height)
            self.color_texture.clear(format="UINT", value=[0x080808])
            self.depth_texture.clear(format="INT", value=[0])

        self.shader.bind()

        # Enable depth test
        gpu.state.depth_test_set("LESS")
        gpu.state.depth_mask_set(True)

        if self.use_atomic_rendering:
            self.shader.image("depth_texture", self.depth_texture)
            self.shader.image("color_texture", self.color_texture)

        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("NONE")

        # get visible objects, this cannot be done in despgraph objects for whatever reason
        hidden_objs = {ob.name for ob in bpy.context.view_layer.objects if not ob.visible_get() and ob.data is not None}

        self.last_used_textures.clear()
        match depsgraph.scene.gameEditorMode:  # game mode implmentations
            case "SM64":
                draw_sm64_scene(self, depsgraph, hidden_objs, space_view_3d, projection_matrix, view_matrix, always_set)
            case "OOT":
                draw_oot_scene(self, depsgraph, hidden_objs, space_view_3d, projection_matrix, view_matrix, always_set)
            case _:
                render_state = get_scene_render_state(depsgraph.scene)
                for obj in depsgraph.objects:
                    obj_info = collect_obj_info(
                        self, obj, depsgraph, hidden_objs, space_view_3d, projection_matrix, view_matrix, always_set
                    )
                    if obj_info is not None:
                        draw_f64_obj(self, render_state, obj_info)

        draw_time = (time.process_time() - t) * 1000
        self.time_total += draw_time
        self.time_count += 1
        # print("Time F3D (ms)", draw_time)

        if self.time_count > 20:
            print("Time F3D AVG (ms)", self.time_total / self.time_count, self.time_count)
            self.time_total = 0
            self.time_count = 0

        if not self.use_atomic_rendering:
            return  # when there's no access to color and depth aux images, we render directly, so skip final 2d draw

        # t = time.process_time()
        gpu.state.face_culling_set("NONE")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS")
        gpu.state.depth_mask_set(False)

        self.init_shader_2d()
        self.shader_2d.bind()

        # @TODO: why can't i cache this?
        vbo_2d = gpu.types.GPUVertBuf(self.shader_2d.format_calc(), 6)
        vbo_2d.attr_fill("pos", [(-1, -1), (-1, 1), (1, 1), (1, 1), (1, -1), (-1, -1)])
        batch_2d = gpu.types.GPUBatch(type="TRIS", buf=vbo_2d)

        self.shader_2d.image("color_texture", self.color_texture)
        batch_2d.draw(self.shader_2d)

        # print("Time 2D (ms)", (time.process_time() - t) * 1000)


class F64RenderSettingsPanel(bpy.types.Panel):
    bl_label = "f64render"
    bl_idname = "OBJECT_PT_F64RENDER_SETTINGS_PANEL"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"

    def draw(self, context):
        f64render_rs: F64RenderSettings = context.scene.f64render.render_settings
        f64render_rs.draw_props(self.layout, context.scene.gameEditorMode)


def draw_render_settings(self, context: bpy.types.Context):
    space_data = context.space_data
    if (
        context.scene.render.engine == Fast64RenderEngine.bl_idname
        and space_data.type == "VIEW_3D"
        and space_data.shading.type in {"MATERIAL", "RENDERED"}
    ):
        self.layout.popover(F64RenderSettingsPanel.bl_idname)


# By default blender will hide quite a few panels like materials or vertex attributes
# Add this method to override the check blender does by render engine
def get_panels():
    exclude_panels = {
        "VIEWLAYER_PT_filter",
        "VIEWLAYER_PT_layer_passes",
    }

    include_panels = {"EEVEE_MATERIAL_PT_context_material", "MATERIAL_PT_preview"}

    panels = []
    for panel in bpy.types.Panel.__subclasses__():
        if hasattr(panel, "COMPAT_ENGINES"):
            if (
                "BLENDER_RENDER" in panel.COMPAT_ENGINES and panel.__name__ not in exclude_panels
            ) or panel.__name__ in include_panels:
                panels.append(panel)

    return panels


def register():
    bpy.types.RenderEngine.f64_render_engine = bpy.props.PointerProperty(type=Fast64RenderEngine)
    for panel in get_panels():
        panel.COMPAT_ENGINES.add("FAST64_RENDER_ENGINE")

    bpy.types.Scene.f64render = bpy.props.PointerProperty(type=F64RenderProperties)

    bpy.types.VIEW3D_HT_header.append(draw_render_settings)

    F64_GLOBALS.clear()


def unregister():
    bpy.types.VIEW3D_HT_header.remove(draw_render_settings)

    del bpy.types.RenderEngine.f64_render_engine

    for panel in get_panels():
        if "FAST64_RENDER_ENGINE" in panel.COMPAT_ENGINES:
            panel.COMPAT_ENGINES.remove("FAST64_RENDER_ENGINE")

    del bpy.types.Scene.f64render
