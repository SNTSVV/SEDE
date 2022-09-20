"""
This module takes care of adding the base mouth shapes to a face rig.

Once this is done, they can be used by the blender-rhubarb addon to lip sync using an audio file.
"""
import bpy
import logging


def bone_position(rig, bone_name: str, rotation_quaternion: list):
    """
    Adjust a bone position in the face rig.

    Parameters
    ----------
    bone_name: str
        The name of the bone whose position is to be modified.
    rotation_quaternion: list
        The rotation parameters specified in quaternion mode.
    """
    # Make sure the rotation parameters have the correct format.
    assert(len(rotation_quaternion) == 4)
    bone = rig.pose.bones[bone_name]
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = rotation_quaternion


def reset_all_bones(rig):
    """
    Reset all bones in the face rig to the initial position.

    Parameters
    ----------
    rig:
        The face rig in which all bone positions shall be reset.
    """
    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = [1.0, 0.0, 0.0, 0.0]


def _select_all_bones_and_add_position(pose_lib, name: str, frame: int):
    """
    Select all bones in the face rig and add the pose to the library.

    Parameters
    ----------
    pose_lib:
        The pose library to which the face position is to be added.
    name: str
        The name of the facial expression to use in the pose library.
    frame: int
        The spot the position shall occupy in the pose library.

    Returns
    -------
        The newly added pose.
    """
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.poselib.pose_add(frame=frame)
    new_pose = pose_lib.pose_markers.get("Pose")
    new_pose.name = name
    return new_pose


def add_base_mouth_shapes_to_rig(rig):
    """
    Add 10 basic mouth shapes used for lip syncing to the face rig.

    To do this, a pose library is created which is then populated with
    the 10 different expressions.

    Parameters
    ----------
    rig:
        The rig to which the basic mouth shape library shall be added.

    Returns
    -------
    tuple
        The first entry is the newly created pose library in the face rig,
        allowing it to be stored and the second one is a dictionary where
        the key labels the mouth shape (e.g. 'A', 'B', 'C', etc.) while the
        value is the corresponding pose. This information can then be used
        to add the poses to the lip sync library (in our case the blender
        rhubarb addon is used.)
    """
    # Start by selecting the whole face rig
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig

    # Go into pose mode
    bpy.ops.object.mode_set(mode="POSE")

    # Create a pose library
    logging.debug("Creating new pose lib for the face rig.")
    bpy.ops.poselib.new()
    bpy.data.actions['PoseLib'].name = "phoneme_basic_mouth_shapes"
    pose_lib = bpy.data.actions["phoneme_basic_mouth_shapes"]

    mouth_shapes = {}

    # Create position A and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_MBP', [0.707, 0.0, 0.0, 0.707])
    mouth_shapes['a'] = _select_all_bones_and_add_position(pose_lib, "A", frame=1)

    # Create position B and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_CDGKNRSYZ', [0.925, 0.0, 0.0, 0.380])
    mouth_shapes['b'] = _select_all_bones_and_add_position(pose_lib, "B", frame=2)

    # Create position C and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_E', [0.707, 0.0, 0.0, 0.707])
    mouth_shapes['c'] = _select_all_bones_and_add_position(pose_lib, "C", frame=3)

    # Create position D and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_AI', [0.814, 0.0, 0.0, 0.581])
    mouth_shapes['d'] = _select_all_bones_and_add_position(pose_lib, "D", frame=4)

    # Create position E and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_O', [0.816, 0.0, 0.0, 0.578])
    mouth_shapes['e'] = _select_all_bones_and_add_position(pose_lib, "E", frame=6)

    # Create position F and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_WQ', [0.891, 0.0, 0.0, 0.453])
    mouth_shapes['f'] = _select_all_bones_and_add_position(pose_lib, "F", frame=7)

    # Create position G and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_FV', [0.822, 0.0, 0.0, 0.569])
    mouth_shapes['g'] = _select_all_bones_and_add_position(pose_lib, "G", frame=8)

    # Create position H and add it to the library
    reset_all_bones(rig)
    bone_position(rig, 'ph_L', [0.707, 0.0, 0.0, 0.707])
    bone_position(rig, 'ph_E', [0.967, 0.0, 0.0, 0.254])
    mouth_shapes['h'] = _select_all_bones_and_add_position(pose_lib, "H", frame=9)

    # Create position H and add it to the library
    reset_all_bones(rig)
    mouth_shapes['x'] = _select_all_bones_and_add_position(pose_lib, "X", frame=10)

    # Go back into object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    return pose_lib, mouth_shapes
