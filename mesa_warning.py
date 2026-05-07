from bpy.types import Context, Event, Operator


class MesaWarningPopup(Operator):
    bl_label = "Mesa Driver Limitations"
    bl_idname = "dialog.f64_mesa_warning"
    bl_description = "Update color management settings to help material preview accuracy"
    bl_options = {"UNDO"}

    already_invoked = False  # HACK: used to prevent multiple dialogs from popping up

    def invoke(self, context: Context, event: Event):
        prefs = context.preferences.addons[__package__].preferences
        if prefs.dont_warn_about_mesa:
            return {"CANCELLED"}
        if MesaWarningPopup.already_invoked:
            return {"FINISHED"}
        MesaWarningPopup.already_invoked = True
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context: Context):
        from fast64_internal.utility import multilineLabel

        col = self.layout.column()
        col.label(text="You are using Mesa drivers!", icon="MEMORY")
        multilineLabel(
            col,
            (
                (
                    "These are much more strict as opposed to commercial drivers\n"
                    "and do not allow us to enable helpful extensions!"
                )
            ),
        )
        col.alert = True
        col.label(text="This will hinder your accuracy and or performance.", icon="ERROR")
        col.alert = False
        multilineLabel(
            col,
            (
                (
                    'You can add "allow_glsl_extension_directive_midshader=true" to your\n'
                    "blender launch arguments to bypass this restriction, or change render\n"
                    "device."
                )
            ),
            icon="INFO",
        )

        prefs = context.preferences.addons[__package__].preferences
        col.prop(prefs, "dont_warn_about_mesa", text="Don't warn me again")

    def cancel(self, context: Context):
        MesaWarningPopup.already_invoked = False

    def execute(self, context):
        return {"FINISHED"}
