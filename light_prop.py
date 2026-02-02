from bpy.types import Context, Panel


def draw_light_color(self: Panel, context: Context):
    if context.scene.render.engine != "FAST64_RENDER_ENGINE":
        return

    layout = self.layout

    layout.use_property_split = True
    layout.use_property_decorate = False

    layout.separator()
    layout.prop(context.light, "color")
