# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "f64render",
    "author": "Max Bebök, Fast64 community",
    "description": "Render engine for fast64 materials.",
    "blender": (3, 2, 0),
    "version": (0, 0, 1),
    "location": "Render Properties > Render Engine",
    "category": "3D View",
}

from .light_prop import draw_light_color
from . import auto_load
import bpy

auto_load.init()


def register():
    bpy.types.DATA_PT_light.append(draw_light_color)
    auto_load.register()


def unregister():
    auto_load.unregister()
    bpy.types.DATA_PT_light.remove(draw_light_color)
