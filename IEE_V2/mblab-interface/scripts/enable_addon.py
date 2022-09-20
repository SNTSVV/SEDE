import bpy

bpy.ops.preferences.addon_enable(module='MB-Lab')
bpy.ops.preferences.addon_enable(module='blender-rhubarb-lipsync')

bpy.ops.wm.save_userpref()
