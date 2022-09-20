"""
Define the landmark labelling of the MB-Lab models.

The labels are universal across models of different sizes,
bodyweight and expressions but differ between male and
female genders.

We consider a subset of 25 landmarks from the ibug 68 standard
labelling scheme, supplement with 2 vertices in the center of the
eyes which are labelled by `69` and `70`.

right eyebow  : 18, 22
left eyebow   : 23, 27
nose ridge    : 28, 31
nose          : 32, 34, 36
right eye     : 37, 38, 39, 40, 41, 42, 69
left eye      : 43, 44, 45, 46, 47, 48, 70
mouth         : 49, 52, 55, 58
"""

import numpy as np
from pathlib import Path
import logging

LABELS_FEMALE = {
    18: 16399,
    22: 16396,
    23: 1310,
    27: 1313,
    28: 7529,
    31: 7573,
    32: 16472,
    34: 7572,
    36: 1386,
    37: 16754,
    38: 17995,
    39: 17066,
    40: 16301,
    41: 18001,
    42: 17988,
    43: 1215,
    44: 2078,
    45: 2077,
    46: 17960,
    47: 17963,
    48: 17976,
    49: 16534,
    52: 7546,
    55: 1464,
    58: 7549,
    69: 6941,
    70: 593,
}

LABELS_MALE = {
    18: 17068,
    22: 16333,
    23: 1929,
    27: 2641,
    28: 7750,
    31: 7762,
    32: 16429,
    34: 7796,
    36: 2002,
    37: 17018,
    38: 16997,
    39: 17002,
    40: 16262,
    41: 17011,
    42: 17758,
    43: 1835,
    44: 2575,
    45: 2570,
    46: 17730,
    47: 17733,
    48: 17746,
    49: 16507,
    52: 7770,
    55: 2080,
    58: 7808,
    69: 7166,
    70: 1214,
}


def store_landmarks(landmarks: dict, filepath: Path, meshname=None):
    """
    Store the landmarks to a file.

    Parameters
    ----------
    landmarks: dict
        Contains the landmark to vertex-id mappings. The landmark indices are the keys,
        the vertex-ids are the values.

    filepath: Path
        The path to the file where the landmark labels shall be stored to.
    meshname: str
        If different from `name`, the meshname will be included in the output
    """
    data = landmarks
    if meshname and 'meshname' not in data:
        data['meshname'] = meshname
    np.save(filepath, data)


def store_landmarks_gender(gender: str, filepath: Path, meshname=None):
    """
    Store the landmarks to a file.

    Parameters
    ----------
    gender: str
        The str indicating whether to store the male or female landmarks to the file.
        Can be either `male` or `female` without any rules on capitalization.
    filepath: Path
        The path to the file where the landmark labels shall be stored to.
    meshname: str
        If different from `name`, the meshname will be included in the output

    Raises
    ------
    ValueError
        In case the provided gender is neither female nor male.
    """
    if not gender.lower() in ['female', 'male']:
        logging.error(f"Got invalid gender: {gender}. Please specify either `male` or `female`")
        raise ValueError(f"Got invalid gender: {gender}. Please specify either `male` or `female`")
    data = LABELS_FEMALE if gender.lower() == 'female' else LABELS_MALE
    if meshname and 'meshname' not in data:
        data['meshname'] = meshname
    np.save(filepath, data)
