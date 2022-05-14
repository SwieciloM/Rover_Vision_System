#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from constants import ARUCO_DICT


def generate_marker(dict_name: str, id: int, side_pixels: int = 420, border_bits: int = 1, disp: bool = True, save: bool = True, path: str = 'images\generated_markers') -> np.ndarray:
    """Creates & saves a canonical marker image.

    Args:
        dict_name (str): Dictionary indicating the type of marker.
        id (int): Valid identifier of the marker that will be returned.
        side_pixels (int, optional): Size of the image in pixels.
        border_bits (int, optional): Width of the marker's border.
        disp (bool, optional): Flag specifying if marker will be displayed.
        save (bool, optional): Flag specifying if marker will be saved on disc.
        path (str, optional): Path to destination where markers will be saved.

    Returns:
        numpy.ndarray: Canonical marker image.

    Raises:
        ValueError: If dictionary is not valid or ID out of range

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
        cv2.imwrite("{}\\Tag no.{} from {}.png".format(path, id, dict_name), tag)

    # Display the generated ArUCo tag on screen
    if disp:
        cv2.imshow("ArUCo Marker", tag)
        cv2.waitKey(0)

    return tag


if __name__ == '__main__':
    # Creating an ArgumentParser object
    ap = argparse.ArgumentParser()

    # Command line arguments:
    ap.add_argument("-t", "--type", type=str, required=True, help="type of ArUCo tag to generate")
    ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
    ap.add_argument("-sp", "--sidepix", type=int, default=420, help="size of the image in pixels")
    ap.add_argument("-bb", "--borderbit", type=int, default=1, help="width of the marker border")
    ap.add_argument("-nd", "--no_disp", action='store_false', default=True, help="flag specifying if marker will be displayed")
    ap.add_argument("-ns", "--no_save", action='store_false', default=True, help="flag specifying if marker will be saved on disc")
    ap.add_argument("-p", "--path", type=str, default='images\generated_markers', help="path to output image containing ArUCo tag")
    args = vars(ap.parse_args())

    # Pass arguments to the function
    generate_marker(args["type"], args["id"], args["sidepix"], args["borderbit"], args["no_disp"], args["no_save"], args["path"])

    # generate_marker('DICT_5X5_250', 10, save=True)
