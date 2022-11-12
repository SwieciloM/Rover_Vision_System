#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse

from typing import Optional
from pathlib import Path
from constants import ARUCO_DICT


def generate_marker(dict_name: str, id: int, side_pixels: int = 420, border_bits: int = 1, disp: bool = True, save: bool = True, path: Optional[str] = None) -> np.ndarray:
    """Creates & saves a canonical marker image.

    Args:
        dict_name (str): Dictionary indicating the type of marker.
        id (int): Valid identifier of the marker that will be returned.
        side_pixels (int, optional): Size of the image in pixels.
        border_bits (int, optional): Width of the marker's border.
        disp (bool, optional): Flag specifying if marker will be displayed.
        save (bool, optional): Flag specifying if marker will be saved on disc.
        path (str, optional): Path to the destination where image is ought to be saved.

    Returns:
        numpy.ndarray: Canonical marker image.

    Raises:
        ValueError: If dictionary is not valid or ID out of range
        SystemError: If image could not be saved

    """
    # Verify that the supplied dict exist and is supported by OpenCV
    if ARUCO_DICT.get(dict_name, None) is None:
        raise ValueError("No such dict_name as '{}'".format(dict_name))

    # Load the ArUCo dict_name
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])

    # Verify that the supplied tag ID exist and is supported by OpenCV
    if id >= len(aruco_dict.bytesList):
        raise ValueError("Tag ID '{}' doesn't exist in {}".format(id, dict_name))

    # Allocate memory for the output ArUCo tag and then draw the ArUCo tag on the output image
    tag = np.zeros((side_pixels, side_pixels, 1), dtype="uint8")
    cv2.aruco.drawMarker(aruco_dict, id, side_pixels, tag, border_bits)

    # Write the generated ArUCo tag to disc
    if save:
        if path is None:
            dest_path = f"Tag no.{id} from {dict_name}.jpg"
        else:
            dest_path = Path(f"{path}")

        saved = cv2.imwrite(str(dest_path), tag)

        if saved:
            print(f"Image saved successfully on {dest_path}")
        else:
            raise SystemError("Could not save image")

    # Display the generated ArUCo tag on screen
    if disp:
        cv2.imshow("ArUCo Marker", tag)
        cv2.waitKey(0)

    return tag


if __name__ == '__main__':
    # Creating an ArgumentParser object
    ap = argparse.ArgumentParser()

    # Function needed for arguments with default NoneType value
    def none_or_str(value):
        return None if value == 'None' else value

    # Command line arguments:
    ap.add_argument("-t", "--type", type=str, required=True, help="type of ArUCo tag to generate")
    ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
    ap.add_argument("-sp", "--sidepix", type=int, default=420, help="size of the image in pixels")
    ap.add_argument("-bb", "--borderbit", type=int, default=1, help="width of the marker border")
    ap.add_argument("-nd", "--no_disp", action='store_false', default=True, help="flag specifying if marker will be displayed")
    ap.add_argument("-ns", "--no_save", action='store_false', default=True, help="flag specifying if marker will be saved on disc")
    ap.add_argument("-p", "--path", type=none_or_str, nargs='?', default=None, help="path to output image containing ArUCo tag")
    args = vars(ap.parse_args())

    # Pass arguments to the function
    generate_marker(args["type"], args["id"], args["sidepix"], args["borderbit"], args["no_disp"], args["no_save"], args["path"])

    # Example:
    # generate_marker('DICT_5X5_100', 80, save=True)
    # aruco_marker_generation.py -t DICT_5X5_100 -i 80
