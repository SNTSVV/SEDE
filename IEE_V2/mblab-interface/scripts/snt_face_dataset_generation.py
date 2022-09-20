import bpy

import argparse
import logging
import random
import shutil
import sys
import os
from math import radians
from pathlib import Path
import numpy as np
# mblab_interface imports

sys.path.append(Path(__file__).parent.parent.__str__() +  '\src')
sys.path.insert(1,'./')
print(sys.path)
import  mblab_interface as mb
import mblab_interface.creation
import mblab_interface.logger
import mblab_interface.label as mbl
import mblab_interface.misc


def setup_cli(args: list):
    """
    Setup the command line interface.
    """
    # fmt: off
    parser = argparse.ArgumentParser(description='Generate humanoid characters using MB-Lab')
    parser.add_argument('data_dir', type=Path, help='Path the root directory of the data storage.')
    parser.add_argument('--output_dir', '-o', type=Path, default='mb-output',
                        help='Specify where to store the output directory.')
    parser.add_argument('--log-level', '-l', choices=['debug', 'info', 'warning', 'error', 'critical'], default='error',
                        help='Specify the log-level. Default: info.')
    parser.add_argument('--mode', '-m', default='w', choices=['w', 'a'],
                        help='Whether to append (`a`) to the log file or overwrite (`w`) the content.')
    parser.add_argument('--log-file', '-f', default=None, type=str,
                        help='File name of the log file to be put in the output directory. '
                        '`None` if no file shall be created.')
    parser.add_argument('--gpu', default="0", type=str, help='Specify which GPUs to use. Example: --gpu 1,2,3,4')
    parser.add_argument('--force', action='store_true', help='Remove the output directory if it exists.')
    parser.add_argument('--render', action='store_true', help='Render the scene.')
    parser.add_argument('--studio-lights', action='store_true', help='Use portrait studio lights.')
    parser.add_argument('--random-engine', type=str, default='Noticeable', choices=['Light', 'Realistic', 'Noticeable', 'Caricature', 'Extreme'],
                        help='Decide how extreme the variations in the randomly generated characters can be.')
    # fmt: on
    return parser.parse_args(args)


def render(path: Path, image_name="render.png"):
    """
    Render the scene and store the rendered image to `path`.
    """
    # Render the scene and save the image
    bpy.data.scenes["Scene"].render.filepath = str(path / image_name)
    # Redirect blender output to file
    bpy.ops.render.render(write_still=True)


def locate_camera(z: float):
    """
    Locate the camera by specifying its z position
    """
    # Position the camera
    # Move the camera
    bpy.data.objects["Camera"].location = [0.0, -0.863247, z]
    bpy.data.objects["Camera"].rotation_euler = [
        radians(90.),
        radians(0.0),
        radians(0.0),
    ]


if __name__ == "__main__":

    # Enable the plugin
    bpy.ops.preferences.addon_enable(module='MB-Lab')

    # Parse the command line arguments
    # Cleanup command line arguments if called through Blender, i.e. ignore Blender command line arguments.
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)
    argv = argv[index:]
    args = setup_cli(argv)

    # Create the output directory
    #output = args.output_dir.resolve()
    #if output.exists() and output.is_dir() and args.force:
    #    shutil.rmtree(output)
    #output.mkdir(parents=True, exist_ok=False)

    # Setup the logger
    mb.logger.setup_logger(
   #     log_level=mb.logger._log_level_from_string(args.log_level),
    )
    #logging.info(f"Storing output to: {output}")

    # GPU setup
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    mb.misc.define_GPU_settings(gpus)

    # Clean the scene
    logging.debug("Cleaning the scene")
    for item in bpy.data.meshes:
        if item.name == "Cube":
            bpy.data.meshes.remove(item)

    # Create random character
    #character = (
    #    mb.creation.HumanCreator.generate_random(
    #        random_engine=args.random_engine,
    #        use_portrait_studio_lights=args.studio_lights,
    #        use_cycles_engine=True,
    #        eye_color_based_on_phenotype=True,                # Choose eye color completely randomly
    #        gender=None                                       # Choose gender randomly
    #    ))

    # Create specific character
    param_vals = os.path.basename(args.output_dir).split("_")
    for j in range(0, len(param_vals)):
        param_vals[j] = float(param_vals[j])
    print("random human")
    print(param_vals)
    character = (
        mb.creation.HumanCreator.generate_human(
            age=float(param_vals[13]),
            eyes_hue=float(param_vals[14]),
            eyes_iris_mix=float(param_vals[15]),
            eyes_sat=float(param_vals[16]),
            eyes_value=float(param_vals[17]),
            skin_freckles=float(param_vals[18]),
            skin_oil=float(param_vals[19]),
            skin_veins=float(param_vals[20]),
            eye_color=float(param_vals[21]),
            random_engine=args.random_engine,
            use_portrait_studio_lights=args.studio_lights,
            use_cycles_engine=True,
            gender=float(param_vals[22]), # None=Choose gender randomly
            randomize=False
        ))
    print("generated")
    humanChar = character
    r = str(random.randint(0, 1000))
    newdir = os.path.join(os.path.dirname(args.output_dir), r)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    #TODO: add eyes_color in creation.py set_eye_color
    character = character.finalize(
            change_object_name_to=f"{r}_obj",                 # Rename the character object
            change_collection_name_to=f"{r}_col",             # Rename the character collection
        )

    # Always add hair (unless due to random choice, bald is selected for men)
    character.add_random_hair(args.data_dir, set_hair_color_based_on_phenotype=True, consider_age=True,
                              grey_hair_minimal_age=0.7, allow_bald_men=True, allow_bald_women=False, influence=70.)

    if random.uniform(0, 1) > 0.15:
        logging.debug("Adding random expression")
        character.set_random_expression(['smile', 'sad', 'angry', 'thinkative'])

    # Relate all the assets to the character so that they can be moved together
    character.combine()

    # Locate the camera by specifying its height depending on the face position
    # Find z-component of vertex 28
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = character._blender_obj.evaluated_get(depsgraph)

    if character.gender == 'female':
        labels = mbl.LABELS_FEMALE
    else:
        labels = mbl.LABELS_MALE

    z = mesh.data.vertices[labels[28]].co.z
    locate_camera(z)

    # Store the blend file with the character
    character.store_blend_file(path=Path(newdir), file_name=r, pack_external=True)
    # Store the landmarks file
    mbl.store_landmarks_gender(character.gender, Path(newdir) / 'landmarks.npy', character._blender_obj.name)

    character = humanChar
    config = {"age": character.age, "eyes_hue": character.eyes_hue, "eyes_iris_mix": character.eyes_iris_mix,
              "eyes_sat": character.eyes_saturation, "eyes_value": character.eyes_value,
              "skin_freckles": character.skin_freckles, "skin_oil": character.skin_oil,
              "skin_veins": character.skin_veins, "gender": character._gender, "race": character._human_type}
    logging.debug(config)
    np.save(os.path.join(newdir,'config.npy'), config)
    # # Render the scene if requested
    #if args.render:
    #    render(Path(newdir), image_name=os.path.basename(args.output_dir) + ".png")
