"""
This module contains the interface to the creation tools in MB-Lab.
"""

import bpy
import math
import logging
import random

from pathlib import Path

from mblab_interface.post_processing import HumanPostProcessor

# The human types that are available and their MB-Lab internal identifiers
_HUMAN_TYPES = {
    "Afro female": "f_af01",
    "Afro male": "m_af01",
    "Asian female": "f_as01",
    "Asian male": "m_as01",
    "Caucasian female": "f_ca01",
    "Caucasian male": "m_ca01",
    "Latino female": "f_la01",
    "Latino male": "m_la01",
}


class HumanCreator:
    """
    This class allows to create humanoids that can later be post processed.

    It allows to set body type as well as phenotypes, skin properties, eye-color or generate them randomly.

    Upon finalization of the model, assets can be added to the body using the class `HumanPostProcessor`.
    """

    # The available eye colors.
    _EYE_COLORS = {"blue": {"RGB": [0.529, 0.639, 0.911, 1.0], "RGB.001": [0.040, 0.156, 0.503, 1.00],
                            "RGB.003": [0.661, 0.952, 1.00, 1.00], "RGB.004": [0.108, 0.362, 0.5, 1.00]},
                   "brown": {"RGB": [0.155, 0.055, 0.013, 1.00], "RGB.001": [0.155, 0.055, 0.013, 1.00],
                             "RGB.002": [0.575, 0.348, 0.188, 1.00], "RGB.003": [0.155, 0.055, 0.013, 1.00],
                             "RGB.004": [0.3, 0.136, 0.078, 1.00]},
                   "green": {"RGB": [0.371, 0.552, 0.642, 1.0], "RGB.001": [0.006, 0.232, 0.029, 1.00],
                             "RGB.002": [0.489, 0.885, 0.270, 1.00], "RGB.003": [0.006, 0.232, 0.029, 1.00]}}

    def __init__(
        self,
        human_type: str,
        use_inverse_kinematic=False,
        use_basic_muscles=False,
        use_cycles_engine=True,
        use_portrait_studio_lights=False,
    ):
        """
        Create a basic human template, that can be edited later.

        Parameters
        ----------
        human_type: str
            Indicates which human type to use for the model.
        use_inverse_kinematics: bool
            Whether or not to use `inverse kinematics`. Defaults to `False`.
        use_basic_muscles: bool
            Whether or not to use `basic muscles`. Defaults to `False`.
        use_cycles_engine: bool
            Whether or not use the `Cycles` engine. If `False`, the `Eevee` engine will be used.
            Defaults to `True`.
        use_portrait_studio_lights: bool
            Whether or not to use studio lights for lightning the human.

        Raises
        ------
        ValueError
            If the `human_type` value is invalid (i.e. if it is not contained in the list `_ALLOWED_HUMAN_TYPES`)
        """
        # Define the human type
        if human_type not in _HUMAN_TYPES:
            raise ValueError(f"Invalid human type. Got {human_type}.")
        else:
            bpy.data.scenes["Scene"].mblab_character_name = _HUMAN_TYPES[human_type]
        logging.debug(f"Creating human with type {_HUMAN_TYPES[human_type]}")

        # Specify whether or not to use inverse kinematics.
        if use_inverse_kinematic:
            bpy.data.scenes["Scene"].mblab_use_ik = True
        else:
            bpy.data.scenes["Scene"].mblab_use_ik = False

        # Specify whether or not to use basic muscles.
        if use_basic_muscles:
            bpy.data.scenes["Scene"].mblab_use_muscle = True
        else:
            bpy.data.scenes["Scene"].mblab_use_muscle = False

        # Specify whether or not to add studio lights.
        if use_portrait_studio_lights:
            bpy.data.scenes["Scene"].mblab_use_lamps = True
        else:
            bpy.data.scenes["Scene"].mblab_use_lamps = False

        # Specify whether to use Cycles or Eevee.
        if use_cycles_engine:
            bpy.data.scenes["Scene"].mblab_use_cycles = True
            bpy.data.scenes["Scene"].mblab_use_eevee = False
        else:
            bpy.data.scenes["Scene"].mblab_use_cycles = False
            bpy.data.scenes["Scene"].mblab_use_eevee = True

        # Create the character.
        bpy.ops.mbast.init_character()

        # Store a reference to the blender object
        self._human_type_code = _HUMAN_TYPES[human_type]
        self._human_type = human_type
        self._blender_obj = bpy.data.objects[self._human_type_code]

        # Look for the iris material
        self._material = None
        for m in bpy.data.materials:
            if "iris" in m.name:
                self._material = m

        # Store the gender
        self._gender = 'female'
        if self._human_type_code.startswith('m'):
            self._gender = 'male'

    def finalize_character_and_backup_images(self, path: Path, filename="human", change_collection_name_to=None,
                                             change_object_name_to=None, add_face_rig=True, add_facs=False):
        """
        Finalize the character an

        Parameters
        ----------
        path: Path
            The path where the backup images shall be stored.
        change_collection_name_to: str
            If requested (not `None`) this will be the new name for the collection containing the character and
            its assets.
        change_object_name_to: str
            If requested (not `None`) this will be the new name for the character object.
        add_face_rig: boolean
            Whether or not to a face rig to the finalzed character (default is `True`).
        add_facs: boolean
            Whether or not to add a facs rig to the finalized character (default is `False`).
        """
        logging.debug(f"Finalizing human model {self._human_type} ({self._human_type_code}) and storing image files.")

        bpy.data.scenes['Scene'].mblab_save_images_and_backup = True
        bpy.data.scenes['Scene'].mblab_export_materials = True
        bpy.data.scenes['Scene'].mblab_export_proportions = True

        if not path.exists():
            path.mkdir(parents=True, exist_ok=False)
        bpy.ops.mbast.export_character(filepath=str(path / (filename + ".json")))
        bpy.ops.mbast.finalize_character_and_images(filepath=str(path / (filename + ".png")), check_existing=True)

        # Identify the new blender object
        blender_obj = None
        blender_col = None
        for ob in bpy.data.objects:
            if "MBlab_bd" in ob.name:
                blender_obj = ob
            if "MBlab_sk" in ob.name:
                blender_col = ob

        if blender_obj is None:
            raise RuntimeError("Could not find the finalized object.")
        if blender_col is None:
            raise RuntimeError("Could not find the finalized collection.")

        if change_collection_name_to is not None:
            blender_col.name = change_collection_name_to
        if change_object_name_to is not None:
            blender_obj.name = change_object_name_to

        toret = HumanPostProcessor(blender_obj, blender_col, self._human_type, gender=self._gender,
                                   age=self._blender_obj.character_age)
        if add_face_rig:
            toret.add_face_rig(import_facs_rig=add_facs)

        return toret

    def finalize(self, change_collection_name_to=None, change_object_name_to=None, add_face_rig=True, add_facs=False):
        """
        Finalize the character an return an object that allows to post-process it.

        Nothing is stored to disk when this method is employed. If you want to store
        the model, please use the method `finalize_character_and_images`.

        Parameters
        ----------
        change_collection_name_to: str
            If requested (not `None`) this will be the new name for the collection containing the character and its
            assets.
        change_object_name_to: str
            If requested (not `None`) this will be the new name for the character object.
        add_face_rig: boolean
            Whether or not to a face rig to the finalzed character (default is `True`).
        add_facs: boolean
            Whether or not to add a facs rig to the finalized character (default is `False`).
        """
        logging.debug(f"Finalizing human model {self._human_type} ({self._human_type_code})")
        bpy.ops.mbast.finalize_character()

        # Identify the new blender object
        blender_obj = None
        blender_col = None
        for ob in bpy.data.objects:
            if "MBlab_bd" in ob.name:
                blender_obj = ob
            if "MBlab_sk" in ob.name:
                blender_col = ob

        if blender_obj is None:
            raise RuntimeError("Could not find the finalized object.")
        if blender_col is None:
            raise RuntimeError("Could not find the finalized collection.")

        if change_collection_name_to is not None:
            blender_col.name = change_collection_name_to
        if change_object_name_to is not None:
            blender_obj.name = change_object_name_to

        toret = HumanPostProcessor(blender_obj, blender_col, self._human_type, gender=self._gender,
                                   age=self._blender_obj.character_age)
        if add_face_rig:
            toret.add_face_rig(import_facs_rig=add_facs)

        return toret

    def reset(self):
        """
        Reset all the changes made to the base model.
        """
        bpy.ops.mbast.reset_allproperties()
        return self

    def set_age(self, age):
        """
        Set the character's age.

        Parameters
        ----------
        age: float
            The variable `age` needs to be in the interval [-1, 1], where
            -1 corresponds to the age of 18 years, while 1 corresponds to 80 years.

        Raises
        ------
        ValueError
            If `age` is not in the interval [-1, 1].
        """
        self.age = age
        if not (age >= -1 and age <= 1):
            raise ValueError(f"Age needs to be in the interval [-1, 1]. Got: {age}")
        else:
            logging.debug(f"Setting character_age = {age}")
            self._blender_obj.character_age = age
            bpy.ops.mbast.skindisplace_calculate()
            return self

    def set_eye_color(self, color: str):
        """
        Set the eye color.

        So far, this is done by choosing from a set of pre-defined colors
        whose settings are stored in the static variable HumanCreator._EYE_COLORS_HSV.

        If an invalid color string is specified, no exception will be thrown.
        Instead a warning is emitted and the execution is continued without color change.

        Parameters
        ----------
        color: str
            The string labelling the requested color.
        """
        self.eyes_color = color
        logging.debug(f"Setting eye color to {color}")
        if color not in HumanCreator._EYE_COLORS:
            logging.warn(f"Requested to change eye color to {color} which is not valid. Not changing eye color.")
            return self
        else:
            color_options = HumanCreator._EYE_COLORS[color]
            for node in color_options:
                self._material.node_tree.nodes[node].outputs[0].default_value = color_options[node]
        return self

    def set_eye_color_based_on_phenotype(self):
        # Note: This case needs to be handled before "asian", otherwise trouble
        if "caucasian" in self._human_type.lower():
            # Allow to keep the default
            if random.choice([True, False]):
                logging.debug("Setting eye color for type caucasian.")
                self.eyes_color = random.choice(["blue", "green"])
                self.set_eye_color(self.eyes_color)
        elif "afro" in self._human_type.lower():
            logging.debug("Setting eye color for type afro.")
            self.set_eye_color("brown")
            self.eyes_color = "brown"
        elif "asian" in self._human_type.lower():
            logging.debug("Setting eye color for type asian.")
            # Keep the default for asian
            pass
        elif "latino" in self._human_type.lower():
            logging.debug("Setting eye color for type latino.")
            self.set_eye_color("brown")
            self.eyes_color = "brown"

        return self

    def set_mass(self, mass):
        """
        Set the character's mass.

        Parameters
        ----------
        mass: float
            The variable `mass` needs to be in the interval [-1, 1], where
            -1 corresponds to the minimal weight necessary to live while 1 corresponds to extreme obesity.

        Raises
        ------
        ValueError
            If `mass` is not in the interval [-1, 1].
        """
        self.mass = mass
        if not (mass >= -1 and mass <= 1):
            raise ValueError(f"Mass needs to be in the interval [-1, 1]. Got: {mass}")
        else:
            logging.debug(f"Setting character_mass = {mass}")
            self._blender_obj.character_mass = mass
            bpy.ops.mbast.skindisplace_calculate()
            return self

    def set_tone(self, tone):
        """
        Set the character's tone.

        Parameters
        ----------
        tone: float
            The variable `tone` needs to be in the interval [-1, 1], where
            -1 corresponds to the mass being mainly due to fat while 1 corresponds to muscle mass.

        Raises
        ------
        ValueError
            If `tone` is not in the interval [-1, 1].
        """
        self.tone = tone
        if not (tone >= -1 and tone <= 1):
            raise ValueError(f"Tone needs to be in the interval [-1, 1]. Got: {tone}")
        else:
            logging.debug(f"Setting character_tone = {tone}")
            self._blender_obj.character_tone = tone
            bpy.ops.mbast.skindisplace_calculate()
            return self

    def randomize_character_properties(
        self,
        random_engine="Noticeable",
        preserve_tone=False,
        preserve_mass=False,
        preserve_body=False,
        preserve_height=False,
        preserve_face=False,
        preserve_phenotype=False,
        preserve_fantasy=True,
    ):
        """
        Randomly modify the character's properties.

        Parameters
        ----------
        random_engine: str
            The random engine to use. Defaults to `Noticeable`.
            Possible choices:
            - Light: Only small changes will be made.
            - Realistic: More severe changes than `Light` but still realistically looking.
            - Noticeable: Produces strong deviations from the original model, while still being realistic.
            - Caricature: Produces models with strong features that are sometimes not realistic anymore.
            - Extreme: Produces models with extreme variations.
        preserve_tone: bool
            Indicate that the tone should not be randomized but kept at the current value.
        preserve_mass: bool
            Indicate that the mass should not be randomized but kept at the current value.
        preserve_body: bool
            Indicate that the body should not be randomized but kept at the current value.
        preserve_height: bool
            Indicate that the height should not be randomized but kept at the current value.
        preserve_face: bool
            Indicate that the face should not be randomized but kept at the current value.
        preserve_phenotype: bool
            Indicate that the phenotype should not be randomized but kept at the current value.
        preserve_fantasy: bool
            Indicate that the fantasy level should not be randomized but kept at the current value.
            This is the only setting that defaults to `False` because it randomization of this value will
            create non-realistic looking human models with pointy ears.
        """
        # Choose the random engine
        if random_engine in [
            "Light",
            "Realistic",
            "Noticeable",
            "Caricature",
            "Extreme",
        ]:
            logging.debug(f"Using random engine: {random_engine}")
            bpy.data.scenes["Scene"].mblab_random_engine = random_engine[0:2].upper()
        else:
            logging.warning(
                f"Got invalid random engine name `{random_engine}`. Using random engine `Noticeable` instead."
            )
            bpy.data.scenes["Scene"].mblab_random_engine = "NO"

        # Specify which properties not to randomize
        if preserve_tone:
            bpy.data.scenes["Scene"].mblab_preserve_tone = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_tone = False

        if preserve_mass:
            bpy.data.scenes["Scene"].mblab_preserve_mass = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_mass = False

        if preserve_body:
            bpy.data.scenes["Scene"].mblab_preserve_body = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_body = False

        if preserve_height:
            bpy.data.scenes["Scene"].mblab_preserve_height = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_height = False

        if preserve_face:
            bpy.data.scenes["Scene"].mblab_preserve_face = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_face = False

        if preserve_phenotype:
            bpy.data.scenes["Scene"].mblab_preserve_phenotype = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_phenotype = False

        if preserve_fantasy:
            bpy.data.scenes["Scene"].mblab_preserve_fantasy = True
        else:
            bpy.data.scenes["Scene"].mblab_preserve_fantasy = False

        # Apply
        bpy.ops.mbast.character_generator()
        return self

    def set_eyes_hue(self, value):
        """
        Set `eyes_hue` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.eyes_hue = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`eyes_hue` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting eyes_hue = {value}")
            self._blender_obj.eyes_hue = value
            return self

    def set_eyes_iris_mix(self, value):
        """
        Set `eyes_iris_mix` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.eyes_iris_mix = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`eyes_iris_mix` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting eyes_iris_mix = {value}")
            self._blender_obj.eyes_iris_mix = value
            return self

    def set_eyes_saturation(self, value):
        """
        Set `eyes_saturation` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.eyes_saturation = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`eyes_saturation` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting eyes_saturation = {value}")
            self._blender_obj.eyes_saturation = value
            return self

    def set_eyes_value(self, value):
        """
        Set `eyes_value` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.eyes_value = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`eyes_value` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting eyes_value = {value}")
            self._blender_obj.eyes_value = value
            return self

    def set_nails_mix(self, value):
        """
        Set `nails_mix` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`nails_mix` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting nails_mix = {value}")
            self._blender_obj.nails_mix = value
            return self

    def set_skin_bump(self, value):
        """
        Set `skin_bump` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`skin_bump` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting skin_bump = {value}")
            self._blender_obj.skin_bump = value
            return self

    def set_skin_complexion(self, value):
        """
        Set `skin_complexion` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`skin_complexion` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting skin_complexion = {value}")
            self._blender_obj.skin_complexion = value
            return self

    def set_skin_freckles(self, value):
        """
        Set `skin_freckles` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.skin_freckles = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`skin_freckles` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting skin_freckles = {value}")
            self._blender_obj.skin_freckles = value
            return self

    def set_skin_oil(self, value):
        """
        Set `skin_oil` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.skin_oil = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`skin_oil` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting skin_oil = {value}")
            self._blender_obj.skin_oil = value
            return self

    def set_skin_veins(self, value):
        """
        Set `skin_veins` to `value`.

        Parameters
        ----------
        value: float
            Needs to be in the interval [0, 1]

        Raises
        ------
        ValueError
            If `value` is not contained in the interval [0, 1]
        """
        self.skin_veins = value
        if not (value >= 0 and value <= 1):
            raise ValueError(f"`skin_veins` value needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting skin_veins = {value}")
            self._blender_obj.skin_veins = value
            return self

    @classmethod
    def generate_human(
            cls,
            age=0,
            eyes_hue=0.5,
            eyes_iris_mix=0.5,
            eyes_sat=0.5,
            eyes_value=0.5,
            skin_freckles=0.5,
            skin_oil=0.5,
            skin_veins=0.5,
            eye_color=None,
            random_engine="Noticeable",
            use_inverse_kinematic=False,
            use_basic_muscles=False,
            use_cycles_engine=True,
            use_portrait_studio_lights=False,
            gender=None,
            randomize=True
    ):
        """
        Create a random character.

        random_engine: str
            The random engine to use. Defaults to `Noticeable`.
            Possible choices:
            - Light: Only small changes will be made.
            - Realistic: More severe changes than `Light` but still realistically looking.
            - Noticeable: Produces strong deviations from the original model, while still being realistic.
            - Caricature: Produces models with strong features that are sometimes not realistic anymore.
            - Extreme: Produces models with extreme variations.
        use_inverse_kinematics: bool
            Whether or not to use `inverse kinematics`. Defaults to `False`.
        use_basic_muscles: bool
            Whether or not to use `basic muscles`. Defaults to `False`.
        use_cycles_engine: bool
            Whether or not use the `Cycles` engine. If `False`, the `Eevee` engine will be used.
            Defaults to `True`.
        use_portrait_studio_lights: bool
            Whether or not to use studio lights for lightning the human.
        eye_color_based_on_phenotype: bool
            Whether or not to choose the eye color based on the phenotype.
        gender: str
            If None, the gender will be chosen randomly. Possible values: `male`, `female`, where
            capitalization does not matter.
        """
        logging.debug("Generating human model.")
        logging.debug("Random:" + str(randomize))
        if gender >= 0.5:
            gender = "female"
        else:
            gender = "male"
        # Choose a human type, taking into account gender if required
        if randomize:
            human_type = random.choice(list(_HUMAN_TYPES.keys()))
        elif gender.lower() == 'female':
            logging.debug("Using fixed gender female in randomly generated character.")
            candidates = filter(lambda s: ' female' in s, _HUMAN_TYPES.keys())
            human_type = random.choice(list(candidates))
        elif gender.lower() == 'male':
            logging.debug("Using fixed gender male in randomly generated character.")
            candidates = filter(lambda s: ' male' in s, _HUMAN_TYPES.keys())
            human_type = random.choice(list(candidates))
        else:
            logging.warning(f"Got invalid gender specifier: {gender}. Ignoring and choosing gender randomly")
            human_type = random.choice(list(_HUMAN_TYPES.keys()))

        character = cls(
            human_type,
            use_inverse_kinematic,
            use_basic_muscles,
            use_cycles_engine,
            use_portrait_studio_lights,
        )

        if randomize:
            # Create a character based on this type
            age = random.uniform(-1.0, 1.0)
            eyes_hue = random.uniform(0., 1.)
            eyes_iris_mix = random.uniform(0., 1.)
            eyes_sat = random.uniform(0., 1.)
            eyes_value = random.uniform(0., 1.)
            skin_freckles = random.uniform(0., 1.)
            skin_oil = random.uniform(0., 1.)
            skin_veins = random.uniform(0., 1.)
            eye_color = random.choice(list(HumanCreator._EYE_COLORS.keys()))
        # Add a random age
        character.set_age(age)

        # Set the properties of the character randomly
        # Here, we preserve the phenotype, as it is already selected randomly, which allows us
        # to set the eyecolor based on the phenotype
        character.randomize_character_properties(random_engine=random_engine, preserve_phenotype=True)

        # Skin edits
        character.set_eyes_hue(eyes_hue)
        character.set_eyes_iris_mix(eyes_iris_mix)
        character.set_eyes_saturation(eyes_sat)
        character.set_eyes_value(eyes_value)
        # character.set_nails_mix(random.uniform(0., 1.))
        # character.set_skin_bump(random.uniform(0., 1.))
        # character.set_skin_complexion(random.uniform(0., 1.))
        character.set_skin_freckles(skin_freckles)
        character.set_skin_oil(skin_oil)
        character.set_skin_veins(skin_veins)

        #if eye_color_based_on_phenotype:
        if randomize:
            logging.debug("Setting eye color based on phenotype in random character generation.")
            character.set_eye_color_based_on_phenotype()
        else:
            logging.debug("NOT setting eye color based on phenotype in random character generation.")
            if eye_color is None:
                eye_color = random.choice(list(HumanCreator._EYE_COLORS.keys()))
            character.set_eye_color(list(HumanCreator._EYE_COLORS.keys())[math.floor(eye_color)])

        return character

    def load_rest_pose(self, path: Path):
        """
        Load a rest pose from a file and apply it.

        Parameters
        ----------
        path: Path
            The path to the `json` file containing the pose information.
        """
        assert(path.suffix == '.json')
        logging.debug(f"Loading rest pose from file: {path}")
        bpy.ops.mbast.restpose_load(filepath=str(path))

        return self
