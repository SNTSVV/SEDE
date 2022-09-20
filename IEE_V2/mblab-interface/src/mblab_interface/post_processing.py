import bpy

import logging
import numpy as np
import os
import random
import subprocess
import wave
import contextlib
import json

from pathlib import Path

from mblab_interface.misc import _save_blend_file
from mblab_interface.logger import log_and_raise as lograise
from mblab_interface.label import LABELS_MALE, LABELS_FEMALE
from mblab_interface.mouth_shapes import add_base_mouth_shapes_to_rig


class HumanPostProcessor:
    """
    This class allows to post process human models.

    An instance of this class is the result of calling one of the finalize
    methods of the `HumanCreator` class.

    Using this class, one can modify a models facial expressions and fit
    assets such as hair, clothes, beards, eyebrows, etc from a base model
    (proxy).
    """

    # A list of supported expressions
    _EXPRESSIONS = ['angry01']
    _MATERIAL_NAME = "mb_lab_interface_hair_material"
    _MELANIN_CONC_TO_COLOR = {
        'white': [0.0, 0.20],
        'blonde': [0.20, 0.40],
        'red': [0.40, 0.65],
        'brown': [0.65, 0.95],
        'black': [0.95, 1.0]
    }
    # list of all pre-defined male poses in mb-lab
    _MALE_POSES = ['captured01', 'evil_beast', 'evil_explains', 'evil_waiting_orders01', 'evil_waiting_orders02',
                   'flying01', 'flying02', 'gym01', 'gym02', 'sit_basic', 'sit_meditation', 'standing_basic',
                   'standing_hero01', 'standing_hero02', 'standing_in_lab', 'standing_old_people',
                   'standing_symmetric']

    # list of all pre-defined female poses in mb-lab
    _FEMALE_POSES = ['captured01', 'flying01', 'flying02', 'glamour01', 'glamour02', 'glamour03', 'glamour04',
                     'glamour05', 'glamour06', 'glamour07', 'glamour08', 'gym01', 'gym02', 'pinup01',
                     'shojo_classic01', 'shojo_classic02', 'shojo_classic03', 'shojo_classic04', 'sit_basic',
                     'sit_meditation', 'sit_sexy', 'sorceress', 'standing01', 'standing02', 'standing03',
                     'standing04', 'standing05', 'standing06', 'standing_basic', 'standing_fitness_competition',
                     'standing_fitness_competition02', 'standing_in_lab', 'standing_old_people', 'standing_symmetric']

    # Correspondence between the MB-Lab bone and the SVIRO bone names
    _SVIRO_BONES_RELABELING = {
        'neck': 'neck_01',
        'spine01': 'spine_01',
        'spine02': 'spine_02',
        'spine03': 'spine_03',
        'thigh_L': 'thigh_l',
        'thigh_R': 'thigh_r',
        'calf_L': 'calf_l',
        'calf_R': 'calf_r',
        'upperarm_L': 'upperarm_l',
        'upperarm_R': 'upperarm_r',
        'lowerarm_L': 'lowerarm_l',
        'lowerarm_R': 'lowerarm_r',
        'hand_L': 'hand_l',
        'hand_R': 'hand_r',
    }

    def __init__(self, blender_obj, blender_col, human_type, gender, age):
        """
        Create a post processor for human models.

        Parameters
        ----------
        blender_obj
            The blender object of the finalized human model.
        blender_coll
            The blender collection, the human model belongs to.
        human_type: str
            The type of the human that can be used to set e.g. its hair color based on the phenotype
        gender: str
            The gender of the model. This information might be needed to choose appropriate hair and clothes
        age: float
            The age (-1, 1) of the model. Can be used to set gray hair.
        """
        logging.debug(f"Creating HumanPostProcessor with name: {blender_obj.name}")
        self._blender_obj = blender_obj
        self._blender_col = blender_col
        self._gender = gender
        self._hair_material = None
        self._principled_hair_bsdf = None
        self._human_type = human_type
        self._age = age
        # Track which bone labelling is being used
        self._bone_labelling = 'default'
        self._face_rig = None
        self._phoneme_rig = None
        self._fit_has_been_performed = False
        self._basic_mouth_shapes = None

    @property
    def gender(self):
        """
        Access the gender of the model.
        """
        return self._gender

    def fit_asset(self, asset: Path, influence: float, offset=0.0, add_mask_group=False):
        """
        Fit the asset located in the file `asset`.

        The threshold is used to indicate whether the asset mesh can be deformed to follow the body mesh.
        For example scalps can be adapted to the model mesh under consideration, while glasses are rigid
        and should use `influence=0`.
        """
        self._fit_has_been_performed = True
        logging.debug(f"Fitting asset {asset}")
        bpy.data.scenes["Scene"].mblab_proxy_library = str(asset.parent)
        bpy.context.scene.mblab_assets_models = asset.stem
        bpy.context.scene.mblab_proxy_name = asset.stem
        # Set a mask group
        if add_mask_group:
            bpy.context.scene.mblab_add_mask_group = True
        bpy.context.scene.mblab_proxy_threshold = influence
        bpy.context.scene.mblab_proxy_offset = offset
        _ = bpy.ops.mbast.proxy_fit()
        if add_mask_group:
            bpy.context.scene.mblab_add_mask_group = False

        return self

    def _get_random_asset_from_dir(self, data_dir: Path):
        """
        Choose a random asset from the directory `data_dir` that is compatible with `self._gender`.
        """
        if not data_dir.is_dir():
            lograise(f"Cannot find directory: {data_dir}", ValueError)
        else:
            candidates = list(data_dir.glob(f"*_{self._gender}*"))
            if len(candidates) == 0:
                logging.warning(f"Did not find any candidates for gender {self._gender} in dir: {data_dir}. "
                                "Not adding any asset.")
                return None
            else:
                return random.choice(candidates)

    def set_random_pose(self):
        """
        Apply a random pose to the model
        """

        # open the context menu of mb-lab responsible for selecting poses
        bpy.ops.mbast.button_pose_on()

        # select the blender object as active
        self._blender_obj.select_set(True)
        bpy.context.view_layer.objects.active = self._blender_obj

        # select randomly a gender
        # there are male and female poses, however, they can be applied to each model
        # but its important to apply it to the correct gender pose below
        random_gender = random.choice(["male", "female"])

        # set the pose depending on the gender
        if random_gender == 'male':

            # select a pose randomly
            random_pose = random.choice(self._MALE_POSES)
            bpy.context.object.male_pose = random_pose

        else:

            # select a pose randomly
            random_pose = random.choice(self._FEMALE_POSES)
            bpy.context.object.female_pose = random_pose

        logging.debug(f"Applying pose: {random_pose}")

        # hacky stuff
        # for some reason, setting the pose only works when using the try except block
        # without it, the poses are never updated
        try:
            bpy.context.object.male_pose = random_pose
        except Exception:
            pass

        try:
            bpy.context.object.female_pose = random_pose
        except Exception:
            pass

    def add_random_glasses(self, data_dir: Path, influence=0.0, offset=0.0, add_mask_group=False,
                           shift=(0.0, 0.0, 0.0)):
        """
        Add random glasses to the model.

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The glasses must be stored under `data_dir/assets/glasses`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).
        shift: tuple
            Shift the mask in the provided direction after it has been fitted.

        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/glasses subdirectory) does not exist.
        """
        logging.debug("Randomly adding glasses to character.")
        glasses = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'glasses')

        # What objects are available before glasses are added.
        available_objs = set([o.name for o in bpy.data.objects])

        if glasses:
            logging.debug(f"Fitting glasses: {glasses}")
            self.fit_asset(glasses, influence, offset, add_mask_group=add_mask_group)

        glasses = None
        new_objects = set([o.name for o in bpy.data.objects]).difference(available_objs)
        assert(len(new_objects) == 1)
        glasses = bpy.data.objects[list(new_objects)[0]]

        if glasses is not None:
            glasses.location.x += shift[0]
            glasses.location.y += shift[1]
            glasses.location.z += shift[2]
            glasses.select_set(True)
            bpy.context.view_layer.objects.active = glasses
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        return self

    def add_random_shoes(self, data_dir: Path, influence=0.0, offset=0.0, add_mask_group=False):
        """
        Add random shoes to the model.

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The shoes must be stored under `data_dir/assets/shoes`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).


        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/shoes subdirectory) does not exist.
        """
        logging.debug("Randomly adding shoes to character.")
        shoes = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'shoes')

        if shoes:
            logging.debug(f"Fitting shoes: {shoes}")
            self.fit_asset(shoes, influence, offset, add_mask_group=add_mask_group)

        return self

    def add_random_mask(self, data_dir: Path, influence=20.0, offset=0.0, add_mask_group=False, shift=(0., 0., 0.)):
        """
        Add random mask to the model.

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The objects must be stored under `data_dir/assets/mask`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).
        shift: tuple
            Shift the mask in the provided direction after it has been fitted.


        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/mask subdirectory) does not exist.
        """
        logging.debug("Randomly adding mask to character.")
        mask = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'mask')

        if mask:
            logging.debug(f"Fitting mask: {mask}")
            self.fit_asset(mask, influence, offset, add_mask_group=add_mask_group)

        mask = None
        for o in bpy.data.objects:
            if 'mask' in o.name.lower():
                mask = o
        if mask is not None:
            mask.location.x += shift[0]
            mask.location.y += shift[1]
            mask.location.z += shift[2]
            mask.select_set(True)
            bpy.context.view_layer.objects.active = mask
            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        return self

    def add_random_clothes_upper_body(self, data_dir: Path, influence=10.0, offset=0.0, add_mask_group=False):
        """
        Add random miscellaneous to the model.

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The objects must be stored under `data_dir/assets/clothes_upper_body`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).


        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/clothes_upper_body subdirectory) does not exist.
        """
        logging.debug("Randomly adding clothes (upper body) to character.")
        clothes = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'clothes_upper_body')

        if clothes:
            logging.debug(f"Fitting upper body clothes: {clothes}")
            self.fit_asset(clothes, influence, offset, add_mask_group=add_mask_group)

        return self

    def add_random_clothes_lower_body(self, data_dir: Path, influence=10.0, offset=0.0, add_mask_group=False):
        """
        Add random clothes to lower body to the model.

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The objects must be stored under `data_dir/assets/clothes_lower_body`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).


        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/clothes_upper_body subdirectory) does not exist.
        """
        logging.debug("Randomly adding clothes (upper body) to character.")
        clothes = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'clothes_lower_body')

        if clothes:
            logging.debug(f"Fitting upper body clothes: {clothes}")
            self.fit_asset(clothes, influence, offset, add_mask_group=add_mask_group)

        return self

    def add_random_clothes_full_body(self, data_dir: Path, influence=10.0, offset=0.0, add_mask_group=False):
        """
        Add random clothes to full body to the model (like a dress or an overall).

        Parameters
        ----------
        data_dir: Path
            The location of the directory containing the data.
            The objects must be stored under `data_dir/assets/clothes_lower_body`.
        influence: float
            The (`magnetic`) attraction the vertices of the character have on the asset to be fitted.
        offset: float
            The offset of the asset vertices to the character vertices.
        add_mask_group: bool
            Whether or not to add a mask group for vertices of the character that are too close to the
            asset vertices and should therefore be hidden (masked).


        Raises
        ------
        FileNotFoundError
            If the data directory, (or the assets/clothes_upper_body subdirectory) does not exist.
        """
        logging.debug("Randomly adding clothes (upper body) to character.")
        clothes = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'clothes_full_body')

        if clothes:
            logging.debug(f"Fitting upper body clothes: {clothes}")
            self.fit_asset(clothes, influence, offset, add_mask_group=add_mask_group)

        return self

    def store_blend_file(self, path: Path, file_name="human", pack_external=False):
        """
        Store the full model in a blend file.

        Parameters
        ----------
        path: Path
            Location of the directory where the blend file shall be stored to.
            If the directory does not exist, it will be created.
        file_name: str
            The name the blend file should have (w/o the `.blend` extension)
        pack_external: bool
            Whether or not to pack external data (texture images) into the blend file.
            Default is `False`. This is useful if the model is moved afterwards.
        """
        if not path.exists():
            path.mkdir(parents=True, exist_ok=False)
        path = path / (file_name + ".blend")
        if path.exists():
            logging.warn(f"Blend file: `{path}` exists and is being overwritten.")
        if pack_external:
            logging.debug("Packing external files in blend file.")
            bpy.ops.file.pack_all()
        _save_blend_file(path)

    def set_angry01(self, value):
        """
        Set the `angry01` expression.

        A `value` of 0 means that the expression is absent (neutral face) while 1 means
        the expression is fully developed.

        Parameters
        ----------
        value: float
            Indicated the prominence of the expression, needs to be in the interval [0, 1]

        Raises
        ------

        """
        if value < 0 or value > 1:
            raise ValueError(f"Value for expression `angry01` needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting value of expression `angry01` to {value}")
            bpy.ops.mbast.button_expressions_on()
            self._blender_obj.angry01 = value
            # bpy.ops.mbast.button_expressions_off()
            return self

    def set_furious03(self, value):
        """
        Set the `furious03` expression.

        A `value` of 0 means that the expression is absent (neutral face) while 1 means
        the expression is fully developed.

        Parameters
        ----------
        value: float
            Indicated the prominence of the expression, needs to be in the interval [0, 1]

        Raises
        ------

        """
        if value < 0 or value > 1:
            raise ValueError(f"Value for expression `furious03` needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting value of expression `furious03` to {value}")
            bpy.ops.mbast.button_expressions_on()
            self._blender_obj.furious03 = value
            # bpy.ops.mbast.button_expressions_off()
            return self

    def set_flirty01(self, value):
        """
        Set the `flirty01` expression.

        A `value` of 0 means that the expression is absent (neutral face) while 1 means
        the expression is fully developed.

        Parameters
        ----------
        value: float
            Indicated the prominence of the expression, needs to be in the interval [0, 1]

        Raises
        ------

        """
        if value < 0 or value > 1:
            raise ValueError(f"Value for expression `flirty01` needs to be in the interval [0, 1]. Got: {value}")
        else:
            logging.debug(f"Setting value of expression `flirty01` to {value}")
            bpy.ops.mbast.button_expressions_on()
            self._blender_obj.flirty01 = value
            # bpy.ops.mbast.button_expressions_off()
            return self

    def create_default_hair_material(self):
        """
        Create hair material for the hair of the character.

        The variable `self._hair_material` takes care of tracking whether a
        default material is already associated to the character. In case
        there is already a material associated to the hair of the character,
        a second call to this method will be ignored and a warning will be
        issued.
        """
        logging.debug(f"Creating hair material {HumanPostProcessor._MATERIAL_NAME}")
        if self._hair_material is not None:
            logging.warning(f"Character {self._blender_obj.name} already has associated hair material. Ignoring call "
                            "to `create_default_hair_material`.")
            return self

        # Use complicated long name by default to avoid collisions
        material_name = HumanPostProcessor._MATERIAL_NAME
        bpy.data.materials.new(name=material_name)
        material = bpy.data.materials[material_name]
        material.use_nodes = True
        # Setup the node system for the material
        node_tree = material.node_tree
        nodes = node_tree.nodes
        # We connect a principled hair bsdf shader to the material output node
        principled_hair_bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
        output_material = nodes['Material Output']

        # Link the nodes
        links = node_tree.links
        links.new(principled_hair_bsdf.outputs['BSDF'], output_material.inputs['Surface'])

        # We use the melanin concentration to define the hair color.
        principled_hair_bsdf.parametrization = 'MELANIN'

        self._hair_material = material
        self._principled_hair_bsdf = principled_hair_bsdf

        return self

    def set_hair_color(self, melanin_concentration=0.5, melanin_redness=0.5):
        """
        Set the hair color by specifying the melanin concentration and the melanin redness.

        Parameters
        ----------
        melanin_concentration: float
            The melanin concentration of the hair which should be between 0 and 1.
        melanin_redness: float
            The melanin redness of the hair which should be between 0 and 1.
        """
        if not self._hair_material:
            self._hair_material = self.create_default_hair_material()
        # Melanin Concentration
        self._principled_hair_bsdf.inputs[1].default_value = melanin_concentration
        # Melanin Redness
        self._principled_hair_bsdf.inputs[2].default_value = melanin_redness

        return self

    def set_hair_color_based_on_phenotype(self, consider_age=True, grey_hair_minimal_age=0.8):
        """
        Set the hair color based on the phenotype.

        This is only done with respect to the melanin concentration.
        The melanin redness will always be chosen randomly.

        Parameters
        ----------
        consider_age: bool
            If character's age is large enough, apply gray hair.
        grey_hair_minimal_age: float
            Minimal age a character needs to have in order be considered obtaining gray hair.
        """
        logging.debug("Setting hair color based on phenotype")
        if not self._hair_material:
            self._hair_material = self.create_default_hair_material()
        # Note: This case needs to be handled before "asian", otherwise trouble
        if "caucasian" in self._human_type.lower():
            allowed_colors = ['blonde', 'red', 'brown', 'black']
            if consider_age and self._age > grey_hair_minimal_age:
                allowed_colors.append('white')
            color = random.choice(allowed_colors)
            color_rng = HumanPostProcessor._MELANIN_CONC_TO_COLOR[color]
            logging.debug(f"Setting hair color for type caucasian to {color}.")
            self.set_hair_color(np.random.uniform(color_rng[0], color_rng[1]),
                                melanin_redness=np.random.uniform(0.0, 1.0))
        elif "afro" in self._human_type.lower():
            allowed_colors = ['brown', 'black']
            if consider_age and self._age > grey_hair_minimal_age:
                allowed_colors.append('white')
            color = random.choice(allowed_colors)
            color_rng = HumanPostProcessor._MELANIN_CONC_TO_COLOR[color]
            logging.debug(f"Setting hair color for type afro to {color}.")
            self.set_hair_color(np.random.uniform(color_rng[0], color_rng[1]),
                                melanin_redness=np.random.uniform(0.0, 1.0))
        elif "asian" in self._human_type.lower():
            allowed_colors = ['brown', 'black']
            if consider_age and self._age > grey_hair_minimal_age:
                allowed_colors.append('white')
            color = random.choice(allowed_colors)
            color_rng = HumanPostProcessor._MELANIN_CONC_TO_COLOR[color]
            logging.debug(f"Setting hair color for type asian to {color}.")
            self.set_hair_color(np.random.uniform(color_rng[0], color_rng[1]),
                                melanin_redness=np.random.uniform(0.0, 1.0))
        elif "latino" in self._human_type.lower():
            allowed_colors = ['brown', 'black']
            if consider_age and self._age > grey_hair_minimal_age:
                allowed_colors.append('white')
            color = random.choice(allowed_colors)
            color_rng = HumanPostProcessor._MELANIN_CONC_TO_COLOR[color]
            logging.debug(f"Setting hair color for type latino to {color}.")
            self.set_hair_color(np.random.uniform(color_rng[0], color_rng[1]),
                                melanin_redness=np.random.uniform(0.0, 1.0))

        return self

    def add_random_hair(self, data_dir: Path, set_hair_color_based_on_phenotype=False, allow_bald_men=False,
                        allow_bald_women=False, consider_age=True, grey_hair_minimal_age=0.8, influence=20.,
                        offset=0.0):
        """
        Add hair

        Parameters
        ----------
        data_dir: Path
            The path to where the data is stored. Hair models are assumed to be located under
            `data_dir` / assets / hair
        set_hair_color_based_on_phenotype: bool
            Whether or not to set the hair color based on the phenotype.

        consider_age: bool
            If character's age is large enough, apply gray hair.
        grey_hair_minimal_age: float
            Minimal age a character needs to have in order be considered obtaining gray hair.
        """
        logging.debug("Adding random hair")
        # Maybe do not add hair
        add_hair = True
        if allow_bald_men and self._gender == 'male':
            add_hair = random.choice([True, False])
        if allow_bald_women and self._gender == 'female':
            add_hair = random.choice([True, False])

        if not add_hair:
            logging.debug("Choosing character to be bald. Not adding hair.")
            return self

        # Add hair material if it is not yet done
        if not self._hair_material:
            self.create_default_hair_material()

        # Define the hair color, either based on phenotype or chosen randomly
        if set_hair_color_based_on_phenotype:
            self.set_hair_color_based_on_phenotype(consider_age=consider_age,
                                                   grey_hair_minimal_age=grey_hair_minimal_age)
        else:
            nodes = bpy.data.materials[HumanPostProcessor._MATERIAL_NAME].node_tree.nodes["Principled Hair BSDF"]
            # Melanin Concentration
            nodes.inputs[1].default_value = np.random.uniform(0.0, 1.0)
            # Melanin Redness
            nodes.inputs[2].default_value = np.random.uniform(0.0, 1.0)

        hair = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'hair')
        if hair:
            logging.debug(f"Fitting hair: {hair}")
            self.fit_asset(hair, influence, offset)

            hair_obj = bpy.data.objects[hair.stem]
            hair_obj.data.materials.clear()
            hair_obj.data.materials.append(self._hair_material)

        return self

    def add_random_beard(self, data_dir: Path, set_hair_color_based_on_phenotype=False, consider_age=True,
                         grey_hair_minimal_age=0.8, influence=500., offset=0.0):
        """
        Add beard

        Parameters
        ----------
        data_dir: Path
            The path to where the data is stored. Hair models are assumed to be located under
            `data_dir` / assets / beard
        set_hair_color_based_on_phenotype: bool
            Whether or not to set the hair color based on the phenotype.

        consider_age: bool
            If character's age is large enough, apply gray hair.
        grey_hair_minimal_age: float
            Minimal age a character needs to have in order be considered obtaining gray hair.
        """
        if self.gender == 'female':
            logging.warning("Adding beard to female character.")

        # Add hair material if it is not yet done
        if not self._hair_material:
            self.create_default_hair_material()

            # Define the hair color, either based on phenotype or chosen randomly
            if set_hair_color_based_on_phenotype:
                self.set_hair_color_based_on_phenotype(consider_age=consider_age,
                                                       grey_hair_minimal_age=grey_hair_minimal_age)
            else:
                # Melanin Concentration
                nodes = bpy.data.materials[HumanPostProcessor._MATERIAL_NAME].node_tree.nodes["Principled Hair BSDF"]
                nodes.inputs[1].default_value = np.random.uniform(0.0, 1.0)
                # Melanin Redness
                nodes.inputs[2].default_value = np.random.uniform(0.0, 1.0)

        beard = self._get_random_asset_from_dir(data_dir.resolve() / 'assets' / 'beard')
        if beard:
            logging.debug(f"Fitting beard: {beard}")
            self.fit_asset(beard, influence, offset)

            beard_obj = bpy.data.objects[beard.stem]
            beard_obj.data.materials.clear()
            beard_obj.data.materials.append(self._hair_material)

        return self

    def add_hair(self, data_hair: Path, set_hair_color_based_on_phenotype=False, consider_age=True,
                 grey_hair_minimal_age=0.8, influence=20., offset=0.5, melanin_concentration=0.5, melanin_redness=0.5):
        """
        Add hair

        Parameters
        ----------
        data_dir: Path
            The path to where the data is stored. Hair models are assumed to be located under
            `data_dir` / assets / hair
        set_hair_color_based_on_phenotype: bool
            Whether or not to set the hair color based on the phenotype.
        hair_name: str
            The name of the hair model to use.
        set_hair_color_based_on_phenotype: bool
            Whether to set the hair color based on the phenotype
        consider_age: bool
            If character's age is large enough, apply gray hair.
        grey_hair_minimal_age: float
            Minimal age a character needs to have in order be considered obtaining gray hair.
        """
        # Add hair material if it is not yet done
        if not self._hair_material:
            self.create_default_hair_material()

        # Define the hair color, either based on phenotype or chosen randomly
        if set_hair_color_based_on_phenotype:
            self.set_hair_color_based_on_phenotype(consider_age=consider_age,
                                                   grey_hair_minimal_age=grey_hair_minimal_age)
        else:
            nodes = bpy.data.materials[HumanPostProcessor._MATERIAL_NAME].node_tree.nodes["Principled Hair BSDF"]
            # Melanin Concentration
            nodes.inputs[1].default_value = melanin_concentration
            # Melanin Redness
            nodes.inputs[2].default_value = melanin_redness

        # Associate the material to the wig

        logging.debug(f"Fitting hair object: {data_hair}")
        self.fit_asset(data_hair, influence=influence, offset=offset)
        # Assign the material
        hair_obj = bpy.data.objects[data_hair.stem]
        hair_obj.data.materials.clear()
        hair_obj.data.materials.append(self._hair_material)

        return self

    def combine(self):
        """
        Combine all the objects in the collection `MB_LAB_Character` to the collection character.
        """
        collection = bpy.data.collections["MB_LAB_Character"]
        for o in collection.all_objects:
            if o is not self._blender_col and o is not self._blender_obj:
                o.parent = self._blender_col
        return self

    def rename_bones(self, rules: dict, bone_label_name: str):
        """
        Rename the bones in the pose of the character according to the `rules`.

        Parameters
        ----------
        rules: dict
            Contains as values the names of the bones to be renamed and as keys their new names.
        bone_label_name: str
            The name of the bone labelling scheme.
        """
        for bone in self._blender_col.pose.bones:
            # Only replace bones that are in the rules
            # TODO: Maybe check that all bones in the rules are present and updated?
            if bone.name in rules:
                logging.debug(f"Relabelling bone with name '{bone.name}' to {rules[bone.name]}")
                self._blender_col.pose.bones[bone.name].name = rules[bone.name]

        self._bone_labelling = bone_label_name
        return self

    def match_mhx2_bone_labelling(self):
        """
        Rename the bones in the pose to match the SVIRO labelling.
        """
        logging.debug("Matching SVIRO bone labelling")
        # We delegate this to another more general method, such that in the future, we can
        # to other relabelings by just providing a dictionary where the keys are the labels
        # of the bones to rename and the values their new names.
        self.rename_bones(HumanPostProcessor._SVIRO_BONES_RELABELING, 'mhx2')

        return self

    def include_face_landmarks(self, col=(1, 0, 0, 1), scale=0.0015):
        """
        Include the facial landmarks.

        Parameters
        ----------
        col: Tuple
            The color of the facial landmark spheres. First three components are the
            rgb values in [0, 1] and the last value is the alpha value.
        """
        # Get the labels corresponding to the gender of the model
        # labels = LABELS_FEMALE if self._gender == 'female' else LABELS_MALE
        if self._gender == 'female':
            logging.debug("Loading female landmarks")
            labels = LABELS_FEMALE
        else:
            logging.debug("Loading male landmarks")
            labels = LABELS_MALE
        # Create the label material
        mat = bpy.data.materials.new(name="MaterialName")
        mat.diffuse_color = col  # change color

        # For each vertex, draw a label
        for label in labels:
            v = self._blender_obj.data.vertices[labels[label]]
            # Create a sphere
            bpy.ops.mesh.primitive_uv_sphere_add(radius=scale, location=v.co, enter_editmode=False)
            s = bpy.data.objects['Sphere']
            s.name = f'sphere_{label}'
            s.data.materials.append(mat)
            # s.parent = self._blender_obj

        return self

    def create_face_rig_if_necessary(foo):
        def magic(self, *args, **kwargs):
            if self._face_rig is None:
                self.add_face_rig()
            foo(self, *args, **kwargs)
        return magic

    def add_face_rig(self, import_facs_rig=False):
        """
        Generate a face rig.

        Parameters
        ----------
        import_facs_rig: bool
            Whether or not to add a facs rig.
        """
        if self._fit_has_been_performed:
            logging.warning("Face rig has to be added before the first asset is fitted!")
            raise RuntimeError("Face rig has to be added before the first asset is fitted!")

        if self._face_rig is not None:
            logging.warning("There is already a face rig associated to the character. Skipping request to create one.")
            return self

        # Construct the face rig
        bpy.data.scenes['Scene'].mblab_facs_rig = import_facs_rig
        bpy.ops.mbast.create_face_rig()
        # Find the object
        for obj in bpy.data.objects:
            if 'face_rig' in obj.name:
                self._face_rig = obj
            if 'phoneme_rig' in obj.name:
                self._phoneme_rig = obj

        # Check if the rigs were found
        if not self._face_rig:
            logging.error('Could not add face rig to character')
            raise RuntimeError('Could add face rig to character')
        if not self._phoneme_rig:
            logging.error('Could not add phoneme rig to character')
            raise RuntimeError('Could not add phoneme rig to character')

        # Rename the rigs
        self._face_rig.name = self._blender_obj.name + '_face_rig'
        self._phoneme_rig.name = self._blender_obj.name + '_phoneme_rig'

        # Hide the rigs during rendering
        for c in bpy.data.collections:
            if "Face_Rig" in c.name:
                c.hide_render = True
            if "Phoneme_Rig" in c.name:
                c.hide_render = True

        return self

    def _set_face_rig_bone(self, bone, quaternion):
        """
        Modify a specifc `bone` of the face rig specified in `quaternion mode`.

        Parameters
        ----------
        bone: str
            The name of the bone to modify.
        quaternion: list
            The rotation parameters specified in quaternion mode.
        """
        assert(len(quaternion) == 4)

        self._face_rig.select_set(True)
        bpy.context.view_layer.objects.active = self._face_rig

        bpy.ops.object.mode_set(mode="POSE")

        # bone = bpy.context.object.pose.bones["Expressions_mouthOpenHalf_max"]
        bone = self._face_rig.pose.bones[bone]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = quaternion
        bpy.ops.object.mode_set(mode="OBJECT")

    ################################################
    # Low-level access to the face rig
    ################################################

    @create_face_rig_if_necessary
    def set_mouth_open_teeth_closed(self, quaternion):
        """
        Define open mouth with teeth closed.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_half to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthOpenTeethClosed_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_open_half(self, quaternion):
        """
        Define how to half-open the mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_half to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthOpenHalf_max", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_open_aggressive(self, quaternion):
        """
        Define how to aggressively open the mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_aggressive to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthOpenAggr_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_open_O(self, quaternion):
        """
        Define how to open the mouth in "O"-shape.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_O to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthOpenO_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_open_lower(self, quaternion):
        """
        Define how to open the lower mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_lower to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthLowerOut_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_inflate(self, quaternion):
        """
        Define how to open the lower mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_inflate to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthInflated_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_horizontal(self, quaternion):
        """
        Define how to move mouth horizontally.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_horizontal to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthHoriz_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_close(self, quaternion):
        """
        Define how to close mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_close to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthClosed_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_chew(self, quaternion):
        """
        Define how mouth chewing position.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_chew to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthChew_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_open_large(self, quaternion):
        """
        Define how to open mouth largely.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_open_large to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthOpenLarge_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_smile_closed(self, quaternion):
        """
        Define how to smile with mouth closed.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_smile_closed to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthSmile_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_smile_open(self, quaternion):
        """
        Define how to smile with mouth open.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_smile_open to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthSmileOpen_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_mouth_smile_open2(self, quaternion):
        """
        Define how to smile with mouth open (second version).

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting mouth_smile_open2 to {quaternion}")
        self._set_face_rig_bone("Expressions_mouthSmileOpen2_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_nostrils_expansion(self, quaternion):
        """
        Set the nostrils expansion.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting nostrils_expansion to {quaternion}")
        self._set_face_rig_bone("Expressions_nostrilsExpansion_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_tongue_horizontal(self, quaternion):
        """
        Set the horizontal tongue position.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting tongue_horizontal to {quaternion}")
        self._set_face_rig_bone("Expressions_tongueHoriz_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_tongue_out_pressure(self, quaternion):
        """
        Set the pressure of the tongue outside the mouth.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting tongue_out_pressure to {quaternion}")
        self._set_face_rig_bone("Expressions_tongueOutPressure_max", quaternion)

    @create_face_rig_if_necessary
    def set_tongue_out(self, quaternion):
        """
        Define how far outside the tongue is.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting tongue_out to {quaternion}")
        self._set_face_rig_bone("Expressions_tongueOut_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_tongue_tip_up(self, quaternion):
        """
        Define how far the tongue tip points upwards.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting tongue_tip_up to {quaternion}")
        self._set_face_rig_bone("Expressions_tongueTipUp_max", quaternion)

    @create_face_rig_if_necessary
    def set_tongue_vertical(self, quaternion):
        """
        Define how far the vertical position of the tongue.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting tongue_vertical to {quaternion}")
        self._set_face_rig_bone("Expressions_tongueVert_max_min", quaternion)

    @create_face_rig_if_necessary
    def set_pupil_dialation(self, quaternion):
        """
        Define the pupil dialation.

        Parameters
        ----------
        quaternion: list
            Contains the rotation of the corresponding bone in quaternion mode.
        """
        logging.debug(f"Setting pupil_dialation to {quaternion}")
        self._set_face_rig_bone("Expressions_pupilsDilatation_max_min", quaternion)

    ################################################
    # High-level access to the face rig
    # Documentation is omitted here.
    # Method names should be self explanatory.
    ################################################

    def set_random_pupil_dialation(self):
        self.set_pupil_dialation([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])

    def set_random_nostrils_expansion(self):
        self.set_nostrils_expansion([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])

    def decorate_with_random_pupil_dialation(foo):
        """
        Decorator to add random pupil dialation whenever calling the method `foo`.
        """
        def magic(self, *args, **kwargs):
            self.set_random_pupil_dialation()
            foo(self, *args, **kwargs)
        return magic

    def decorate_with_random_nostrils_expansion(foo):
        """
        Decorator to add random nostrils expansion whenever calling the method `foo`.
        """
        def magic(self, *args, **kwargs):
            self.set_random_nostrils_expansion()
            foo(self, *args, **kwargs)
        return magic

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_smile(self):
        logging.debug("Setting random smile")
        # Randomly choose smile type
        smile_types = ["closed", "open_1", "open_2"]
        smile = random.choice(smile_types)
        if smile == "closed":
            self.set_mouth_smile_closed([0.960, random.uniform(-0.276, 0.0), 0.0, 0.0])
        elif smile == "open_1":
            self.set_mouth_smile_open([0.960, random.uniform(-0.276, 0.0), 0.0, 0.0])
        else:
            self.set_mouth_smile_open2([0.960, random.uniform(-0.276, 0.0), 0.0, 0.0])

        # If the mouth is open, do something to the tongue
        # if smile in smile_types[1:]:
        #     self.set_tongue_horizontal([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])
        #     self.set_tongue_out([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])
        #     self.set_tongue_out_pressure([0.960, random.uniform(-0.276, 0.0), 0.0, 0.0])
        #     self.set_tongue_tip_up([0.960, random.uniform(-0.276, 0.0), 0.0, 0.0])
        #     self.set_tongue_vertical([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_sad(self):
        logging.debug("Setting random sad expression")
        # Randomly choose smile type (programmatically this corresponds to an inverted smile)
        smile_types = ["closed", "open_1", "open_2"]
        smile = random.choice(smile_types)
        if smile == "closed":
            self.set_mouth_smile_closed([0.960, random.uniform(0., 0.279), 0.0, 0.0])
            self.set_mouth_close([0.960, random.uniform(0., 0.279), 0.0, 0.0])
        elif smile == "open_1":
            self.set_mouth_smile_open([0.960, random.uniform(0, 0.279), 0.0, 0.0])
        else:
            self.set_mouth_smile_open2([0.960, random.uniform(0, 0.279), 0.0, 0.0])

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_angry(self):
        logging.debug("Setting random angry expression")

        angry_types = ["angry", "furious"]
        angry = random.choice(angry_types)

        if angry == "angry":
            self.set_mouth_open_teeth_closed([0.960, random.uniform(-0.276, 0.111), 0.0, 0.0])
        else:
            self.set_mouth_open_aggressive([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_thinkative(self):
        logging.debug("Setting random thinkative expression")

        think_types = ["left", "left"]
        think = random.choice(think_types)

        if think == "left":
            self.set_mouth_horizontal([0.960, random.uniform(-0.276, 0.111), 0.0, 0.0])
        else:
            self.set_mouth_horizontal([0.960, random.uniform(0.111, 0.279), 0.0, 0.0])

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_tongue_out(self):
        logging.debug("Setting random tongue-out expression")

        if random.choice([True, False]):
            self.set_mouth_open_O([0.960, random.uniform(-0.276, -0.206), 0.0, 0.0])
        else:
            self.set_mouth_open_half([0.960, random.uniform(-0.276, -0.206), 0.0, 0.0])

        self.set_tongue_out([0.960, -0.279, 0.0, 0.0])
        self.set_tongue_vertical([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])
        self.set_tongue_tip_up([0.960, random.uniform(-0.276, 0.), 0.0, 0.0])
        self.set_tongue_horizontal([0.960, random.uniform(-0.276, 0.279), 0.0, 0.0])
        self.set_tongue_out_pressure([0.960, random.uniform(-0.276, 0.), 0.0, 0.0])

    @decorate_with_random_nostrils_expansion
    @decorate_with_random_pupil_dialation
    def set_random_expression(self, generator_flags=['smile', 'sad', 'angry', 'thinkative', 'tongue']):
        generators = []

        if 'smile' in generator_flags:
            generators.append(self.set_random_smile)
        if 'sad' in generator_flags:
            generators.append(self.set_random_sad)
        if 'angry' in generator_flags:
            generators.append(self.set_random_angry)
        if 'thinkative' in generator_flags:
            generators.append(self.set_random_thinkative)
        if 'tongue' in generator_flags:
            generators.append(self.set_random_tongue_out)

        expression_generator = random.choice(generators)
        expression_generator()

    def _add_base_mouth_shapes(self):
        """
        Add 10 basic mouth shapes used for lip syncing to the facerig.

        Raises
        ------
        RuntimeError
            If no face rig is associated to the character.
        """
        if not self._phoneme_rig:
            logging.error("Trying to add basic mouth shapes but there is no face rig.")
            raise RuntimeError("Trying to add basic mouth shapes but there is no face rig.")

        logging.debug("Adding base mouth shapes to rig.")
        rig = self._phoneme_rig
        pose_lib, mouth_shapes = add_base_mouth_shapes_to_rig(rig)

        self._basic_mouth_shapes = pose_lib

        # Add the mouth shapes to rhubarb
        pose_lib.mouth_shapes.mouth_a = mouth_shapes['a'].name
        pose_lib.mouth_shapes.mouth_b = mouth_shapes['b'].name
        pose_lib.mouth_shapes.mouth_c = mouth_shapes['c'].name
        pose_lib.mouth_shapes.mouth_d = mouth_shapes['d'].name
        pose_lib.mouth_shapes.mouth_e = mouth_shapes['e'].name
        pose_lib.mouth_shapes.mouth_f = mouth_shapes['f'].name
        pose_lib.mouth_shapes.mouth_g = mouth_shapes['g'].name
        pose_lib.mouth_shapes.mouth_h = mouth_shapes['h'].name
        pose_lib.mouth_shapes.mouth_x = mouth_shapes['x'].name

    def _set_keyframes(self, context, frame):
        for bone in context.selected_pose_bones:
            bone.keyframe_insert(data_path='location', frame=frame)
            if bone.rotation_mode == 'QUATERNION':
                bone.keyframe_insert(data_path='rotation_quaternion', frame=frame)
            else:
                bone.keyframe_insert(data_path='rotation_euler', frame=frame)
            bone.keyframe_insert(data_path='scale', frame=frame)

    def lip_sync(self, audio_file: Path, script_file: Path, render_animation_to=None, start_frame=1,
                 frame_rate=24, hold_frame_threshold=4):
        """
        Perform lip syncing for the model by specifying an audio file and a script file.

        Parameters
        ----------
        audio: Path
            The path to the audio file that is to be used for lip syncing.
        script_file: Path
            Path to the script file. This is a simple `.txt` file that contains a
            scripted version of what is spoken in the audio file to aide the lip
            syncing algorithms in `rhubarb`.
        start_frame: int
            The frame at which the generated handles for the face rig that perform
            the lip syncing through moving the bones shall be put. This indicated the
            frame at which the character starts to speak. (Obviously the beginning of
            the audio file needs to be put in the same frame to have the audio nicely
            synced to the video)
        render_andimation_to: Path
            Path where to store the (muted) animation rendered as a sequence of PNG's.
        frame_rate: float
            The frame rate to use for rendering
        hold_frame_threshold: int
            Hold key to add if time since large key is large

        Raises
        ------
        RuntimeError
            If no face rig is associated to the character.
        FileNotFoundError
            If either the audio file or the script file are not found.
            If the render directory does not exist.
        """
        logging.debug("calling PostProcessor.lip_sync")

        # Start off by adding basic mouth shapes to the face rig if this has not been done yet.
        if not self._basic_mouth_shapes:
            self._add_base_mouth_shapes()

        # Make sure, the audio and script files exist
        if not audio_file.is_file():
            logging.error(f"Audio file '{audio_file}' does not exist")
            raise FileNotFoundError(f"Audio file '{audio_file}' does not exist")
        if not script_file.is_file():
            logging.error(f"Script file '{script_file}' does not exist")
            raise FileNotFoundError(f"Script file '{script_file}' does not exist")

        # The starting frame cannot be negarive
        if start_frame < 0:
            logging.warning("Got negative starting frame for lip syncing. Using start frame 0 instead.")
            start_frame = 0

        # Load the audio and script files into the blender rhubarb addon.
        self._basic_mouth_shapes.mouth_shapes.sound_file = str(audio_file)
        self._basic_mouth_shapes.mouth_shapes.dialog_file = str(script_file)

        # Set the starting frame
        self._basic_mouth_shapes.mouth_shapes.start_frame = start_frame

        # Set the starting frame
        self._basic_mouth_shapes.mouth_shapes.start_frame = start_frame

        # Set the frame rate
        bpy.context.scene.render.fps = frame_rate

        # Perform the lip syncing
        self._phoneme_rig.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")

        # Repeat what is done in the rhubarb blender pluging at
        # https://github.com/scaredyfish/blender-rhubarb-lipsync/blob/master/op_blender_rhubarb.py
        # as suggested to me in
        # https://github.com/scaredyfish/blender-rhubarb-lipsync/issues/19#issuecomment-673763787
        context = bpy.context
        preferences = context.preferences
        addon_prefs = preferences.addons["blender-rhubarb-lipsync"].preferences

        inputfile = bpy.path.abspath(context.object.pose_library.mouth_shapes.sound_file)
        dialogfile = bpy.path.abspath(context.object.pose_library.mouth_shapes.dialog_file)
        recognizer = bpy.path.abspath(addon_prefs.recognizer)
        executable = bpy.path.abspath(addon_prefs.executable_path)

        exe_path = Path(executable)
        if not exe_path.is_file():
            raise FileNotFoundError(f"Cannot find rhubarb executable under: {exe_path}")

        # This is ugly, but Blender unpacks the zip without execute permission
        os.chmod(executable, 0o744)

        command = [executable, "-f", "json", "--machineReadable", "--extendedShapes", "GHX", "-r",
                   recognizer, inputfile]

        if dialogfile:
            command.append("--dialogFile")
            command.append(dialogfile)

        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)

        results = json.loads(output.stdout)

        lib = self._basic_mouth_shapes
        last_frame = 0
        prev_pose = 0

        for cue in results['mouthCues']:
            frame_num = round(cue['start'] * frame_rate) + lib.mouth_shapes.start_frame

            # add hold key if time since last key is large
            if frame_num - last_frame > hold_frame_threshold:
                logging.debug(f"lip-sync: hold frame: {frame_num - hold_frame_threshold}")
                bpy.ops.poselib.apply_pose(pose_index=prev_pose)
                self._set_keyframes(bpy.context, frame_num - hold_frame_threshold)

            logging.debug("lip-sync: start: {0} frame: {1} value: {2}".format(cue['start'], frame_num,
                          cue['value']))

            mouth_shape = 'mouth_' + cue['value'].lower()
            if mouth_shape in context.object.pose_library.mouth_shapes:
                pose_index = context.object.pose_library.mouth_shapes[mouth_shape]
            else:
                pose_index = 0

            bpy.ops.poselib.apply_pose(pose_index=pose_index)
            self._set_keyframes(bpy.context, frame_num)

            prev_pose = pose_index
            last_frame = frame_num

        # Check how long the audio file is
        # https://stackoverflow.com/questions/7833807/get-wav-file-length-or-duration
        with contextlib.closing(wave.open(str(audio_file), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = round(frames / float(rate))

        # Compute the number of frames we need based on the frame rate
        number_of_frames = frame_rate * duration + start_frame
        bpy.data.scenes['Scene'].frame_end = number_of_frames
