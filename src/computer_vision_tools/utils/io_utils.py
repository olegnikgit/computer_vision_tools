#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of utility functions for input/output operations.

@author: ons
"""
# =============================================================================
import yaml
import os
from importlib import resources
import numpy as np
import re
import glob
from difflib import SequenceMatcher


# =============================================================================
def load_user_config(user_config_file):
    """Load user configuration."""

    if user_config_file is None:
        print("User configuration is not given.")
        return None

    # 1. Direct file path
    if os.path.exists(user_config_file):
        with open(user_config_file, "r") as f:
            return yaml.safe_load(f)

    # 2. Try loading from package resources
    try:
        if not user_config_file.endswith(".yaml"):
            user_config_file += ".yaml"

        package = "r3_mydas_battery_demo.config.users"

        with resources.files(package).joinpath(user_config_file).open("r") as f:
            return yaml.safe_load(f)

    except FileNotFoundError:
        print(f"User configuration file '{user_config_file}' does not exist.")
        return None


def save_calibration(filename, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Save camera calibration parameters to a .npz file.

    :param filename: Output file path (e.g., 'calibration_data.npz')
    :param camera_matrix: Camera intrinsic matrix (3x3)
    :param dist_coeffs: Distortion coefficients (k1, k2, p1, p2, etc.)
    :param rvecs: List of rotation vectors
    :param tvecs: List of translation vectors
    """
    np.savez(
        filename,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object)
    )


def load_calibration(filename):
    """
    Load camera calibration parameters from a .npz file.

    :param filename: Path to the saved .npz file
    :return: Tuple of (camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    data = np.load(filename, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    return camera_matrix, dist_coeffs, rvecs, tvecs


def natural_sort_key(s):
    """
    Natural sorting key for sorting strings with numbers in extracted from the string.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_images(path):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path, ext)))
    return sorted(files)


def _common_substring(strings):
    """
    Finds the longest common substring across a list of strings.
    """
    if not strings:
        return ""

    def lcs(a, b):
        match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
        return a[match.a: match.a + match.size]

    common = strings[0]
    for s in strings[1:]:
        common = lcs(common, s)
        if not common:
            break
    return common



def get_image_pairs(left_path, right_path):
    left_files = glob.glob(os.path.join(left_path, "*"))
    right_files = glob.glob(os.path.join(right_path, "*"))

    left_names = [os.path.basename(f) for f in left_files]
    right_names = [os.path.basename(f) for f in right_files]

    # Find common substring in filenames
    common_left = _common_substring(left_names)
    common_right = _common_substring(right_names)

    def key_func_left(path):
        name = os.path.basename(path)
        core = name.replace(common_left, "")
        return core

    def key_func_right(path):
        name = os.path.basename(path)
        core = name.replace(common_right, "")
        return core

    # Sort by "core name" (filename minus common string)
    left_sorted = sorted(left_files, key=key_func_left)
    right_sorted = sorted(right_files, key=key_func_right)

    # Pair by sorted order
    pairs = []
    for lf, rf in zip(left_sorted, right_sorted):
        if os.path.exists(rf):
            pairs.append((lf, rf))
        else:
            print(f"Missing pair for {lf}")

    # check if pairs are correctly matched by comparing their core names:
    for lf, rf in pairs:
        core_l = os.path.basename(lf).replace(common_left, "")
        core_r = os.path.basename(rf).replace(common_right, "")
        if core_l != core_r:
            # throw an error if they don't match:
            raise ValueError(f"Filename mismatch: {lf} vs {rf}")
        else:
            print(f"Matched pair: {lf} <-> {rf}")

    return pairs