import bpy
from bpy.types import PropertyGroup, Image

from .globals import F64_GLOBALS

# TODO: Some things are from fast64 but can´t be imported at runtime

enumTexFormat = [
    ("I4", "Intensity 4-bit", "Intensity 4-bit"),
    ("I8", "Intensity 8-bit", "Intensity 8-bit"),
    ("IA4", "Intensity Alpha 4-bit", "Intensity Alpha 4-bit"),
    ("IA8", "Intensity Alpha 8-bit", "Intensity Alpha 8-bit"),
    ("IA16", "Intensity Alpha 16-bit", "Intensity Alpha 16-bit"),
    ("CI4", "Color Index 4-bit", "Color Index 4-bit"),
    ("CI8", "Color Index 8-bit", "Color Index 8-bit"),
    ("RGBA16", "RGBA 16-bit", "RGBA 16-bit"),
    ("RGBA32", "RGBA 32-bit", "RGBA 32-bit"),
    # ('YUV16','YUV 16-bit', 'YUV 16-bit'),
]

enumCIFormat = [
    ("RGBA16", "RGBA 16-bit", "RGBA 16-bit"),
    ("IA16", "Intensity Alpha 16-bit", "Intensity Alpha 16-bit"),
]


def simplified_tex_update(self, context):
    from fast64_internal.f3d.f3d_material import setAutoProp

    tex_size = self.get_tex_size()
    if self.tex is not None and self.autoprop:
        setAutoProp(self.S, tex_size[0])
        setAutoProp(self.T, tex_size[1])


class TextureFieldProperty(PropertyGroup):
    clamp: bpy.props.BoolProperty(name="Clamp")
    mirror: bpy.props.BoolProperty(name="Mirror")
    low: bpy.props.FloatProperty(
        name="Low",
        min=0,
        max=1023.75,
    )
    high: bpy.props.FloatProperty(
        name="High",
        min=0,
        max=1023.75,
    )
    mask: bpy.props.IntProperty(
        name="Mask",
        min=0,
        max=15,
        default=5,
    )
    shift: bpy.props.IntProperty(
        name="Shift",
        min=-5,
        max=10,
    )


class TextureProperty(PropertyGroup):
    tex: bpy.props.PointerProperty(
        type=Image,
        name="Texture",
        update=simplified_tex_update,
    )

    tex_format: bpy.props.EnumProperty(
        name="Format",
        items=enumTexFormat,
        default="RGBA16",
        update=simplified_tex_update,
    )
    ci_format: bpy.props.EnumProperty(
        name="CI Format",
        items=enumCIFormat,
        default="RGBA16",
        update=simplified_tex_update,
    )
    S: bpy.props.PointerProperty(type=TextureFieldProperty)
    T: bpy.props.PointerProperty(type=TextureFieldProperty)

    menu: bpy.props.BoolProperty()
    autoprop: bpy.props.BoolProperty(
        name="Autoprop",
        default=True,
        update=simplified_tex_update,
    )

    def get_tex_size(self) -> list[int]:
        if self.tex:
            return self.tex.size
        return [0, 0]

    def draw_default_ui(self, layout: bpy.types.UILayout, index: int):
        def small_split(layout, prop: str, name: str):
            split = layout.split(factor=0.25)
            split.label(text=name)
            split.prop(self, prop, text="")

        def s_t(layout, name: str):
            split = layout.split(factor=0.25)
            split.label(text=name)
            row = split.row()
            row.prop(self.S, name.lower(), text="S")
            row.prop(self.T, name.lower(), text="T")

        col = layout.column()
        row = col.row()
        row.alignment = "LEFT"
        row.prop(self, "menu", text="Texture " + str(index), icon="TRIA_DOWN" if self.menu else "TRIA_RIGHT")
        if self.menu:
            col.template_ID(self, "tex", new="image.new", open="image.open")
            small_split(col, "tex_format", "Format")
            if self.tex_format.startswith("CI"):
                small_split(col, "ci_format", "")
            s_t(col, "Clamp")
            s_t(col, "Mirror")
            col.prop(self, "autoprop", text="Auto Properties")
            if not self.autoprop:
                s_t(col, "Low")
                s_t(col, "High")
                s_t(col, "Mask")
                s_t(col, "Shift")


def update_all_materials(_scene, _context):
    F64_GLOBALS.materials_cache = {}


def rebuild_shaders(_scene, _context):
    F64_GLOBALS.rebuid_shaders = True


class F64RenderSettings(bpy.types.PropertyGroup):
    use_atomic_rendering: bpy.props.BoolProperty(
        name="Use Atomic Rendering",
        default=True,
        description="Atomic rendering will draw to a depth and color buffer seperately, which allows for proper blender and decal emulation.\n"
        "This may cause artifacts if your GPU does not support the interlock extension",
        update=rebuild_shaders,
    )
    sources_tab: bpy.props.BoolProperty(name="Default Sources")
    default_prim_color: bpy.props.FloatVectorProperty(
        description="Primitive Color",
        default=(1, 1, 1, 1),
        subtype="COLOR",
        size=4,
        min=0,
        max=1,
    )
    default_lod_frac: bpy.props.FloatProperty(
        description="Prim LOD Frac",
        min=0,
        max=1,
        step=1,
    )
    default_lod_min: bpy.props.FloatProperty(
        description="Min LOD Ratio",
        min=0,
        max=1,
        step=1,
    )
    default_env_color: bpy.props.FloatVectorProperty(
        description="Enviroment Color",
        default=(0.5, 0.5, 0.5, 0.5),
        subtype="COLOR",
        size=4,
        min=0,
        max=1,
    )

    chroma_tab: bpy.props.BoolProperty(name="Chroma Key")
    default_key_center: bpy.props.FloatVectorProperty(
        description="Key Center",
        subtype="COLOR",
        size=4,
        min=0,
        max=1,
        default=(1, 1, 1, 1),
    )
    default_key_scale: bpy.props.FloatVectorProperty(
        description="Key Scale",
        min=0,
        max=1,
        step=1,
    )
    default_key_width: bpy.props.FloatVectorProperty(
        description="Key Width",
        min=0,
        max=16,
    )

    convert_tab: bpy.props.BoolProperty(name="YUV Convert")
    default_convert: bpy.props.FloatVectorProperty(
        description="YUV Convert",
        size=6,
        min=-1,
        max=1,
        step=1,
        default=(175 / 255, -43 / 255, -89 / 255, 222 / 255, 114 / 255, 42 / 255),
    )

    texture_tab: bpy.props.BoolProperty(name="Textures")
    default_tex0: bpy.props.PointerProperty(type=TextureProperty)
    default_tex1: bpy.props.PointerProperty(type=TextureProperty)
    default_tex2: bpy.props.PointerProperty(type=TextureProperty)
    default_tex3: bpy.props.PointerProperty(type=TextureProperty)
    default_tex4: bpy.props.PointerProperty(type=TextureProperty)
    default_tex5: bpy.props.PointerProperty(type=TextureProperty)
    default_tex6: bpy.props.PointerProperty(type=TextureProperty)
    default_tex7: bpy.props.PointerProperty(type=TextureProperty)
    always_set: bpy.props.BoolProperty(name='Ignore "Set (Source)"', update=update_all_materials)

    render_type: bpy.props.EnumProperty(
        items=[
            ("DEFAULT", "Always Draw", "Always Draw"),
            ("IGNORE", 'Respect "Ignore Render"', 'Respect "Ignore Render"'),
            ("COLLISION", "Only Collision", "Collision"),
        ],
    )
    sm64_specific_area: bpy.props.PointerProperty(
        type=bpy.types.Object, poll=lambda self, obj: obj.type == "EMPTY" and obj.sm64_obj_type == "Area Root"
    )
    oot_specific_room: bpy.props.PointerProperty(
        type=bpy.types.Object, poll=lambda self, obj: obj.type == "EMPTY" and obj.ootEmptyType == "Room"
    )

    def draw_props(self, layout: bpy.types.UILayout, gameEditorMode: str):
        from fast64_internal.utility import prop_split, multilineLabel

        layout = layout.column()
        if gameEditorMode in {"SM64", "OOT"}:
            prop_split(layout, self, "render_type", "Render Type")
            if gameEditorMode == "SM64":
                prop_split(layout, self, "sm64_specific_area", "Specific Area")
            if gameEditorMode == "OOT":
                prop_split(layout, self, "oot_specific_room", "Specific Room")
            layout.separator()

        if bpy.app.version >= (4, 1, 0):
            layout.prop(self, "use_atomic_rendering")
        layout.prop(self, "always_set")
        layout.prop(self, "sources_tab", icon="TRIA_DOWN" if self.sources_tab else "TRIA_RIGHT")
        if self.sources_tab:
            sources_box = layout.box().column()
            prop_split(sources_box, self, "default_prim_color", "Primitive")
            prim_lod_row = sources_box.row()
            prim_lod_row.prop(self, "default_lod_frac", text="Frac")
            prim_lod_row.prop(self, "default_lod_min", text="Min")
            prop_split(sources_box, self, "default_env_color", "Environment")

            sources_box.prop(self, "chroma_tab", icon="TRIA_DOWN" if self.chroma_tab else "TRIA_RIGHT")
            if self.chroma_tab:
                prop_split(sources_box, self, "default_key_center", "Center")
                prop_split(sources_box, self, "default_key_scale", "Scale")
                prop_split(sources_box, self, "default_key_width", "Width")
                if self.default_key_width[0] > 1 or self.default_key_width[1] > 1 or self.default_key_width[2] > 1:
                    multilineLabel(sources_box.box(), "Keying is disabled for\nchannels with width > 1.", icon="INFO")

            sources_box.prop(self, "convert_tab", icon="TRIA_DOWN" if self.convert_tab else "TRIA_RIGHT")
            if self.convert_tab:
                sources_box.prop(self, "default_convert", text="")

            sources_box.prop(self, "texture_tab", icon="TRIA_DOWN" if self.texture_tab else "TRIA_RIGHT")
            if self.texture_tab:
                texture_box = sources_box.box().column()
                for i in range(8):
                    tex_prop = getattr(self, f"default_tex{i}")
                    tex_prop.draw_default_ui(texture_box, i)


class F64RenderProperties(bpy.types.PropertyGroup):
    render_settings: bpy.props.PointerProperty(type=F64RenderSettings)
