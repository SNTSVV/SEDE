import bpy

import logging

from pathlib import Path


def _save_blend_file(path: Path):
    """
    Save a blender scene to a `.blend` file.

    Parameters
    ----------
    path: Path
        The path to the output file.
    """
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        logging.info(f"Saved scene to: {path}")
    except Exception as e:
        logging.error(f"Unable to save scene to: {path}")
        logging.error(f"{e}")
        # Rethrow exception
        raise e


def define_GPU_settings(which_gpus: str):
    logging.debug(f"Setting GPU: {which_gpus}")
    # specify in the scene to use gpu and cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # get the settings (user preferences)
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # specify to use CUDA
    prefs.compute_device_type = 'CUDA'

    # check the actual devices installed (the above is only the ones in the preferences)
    cuda_devices, _ = prefs.get_devices()

    # do not consider the last one (i.e. the CPU)
    for i in range(len(cuda_devices)-1):
        cuda_devices[i].use = True if i in which_gpus else False

    # save the settings to the profile
    bpy.ops.wm.save_userpref()
