#!/usr/bin/python
# -*- coding: utf-8 -*-

# opencv-contrib-python
import cv2
import numpy as np
import argparse
import sys


# Define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# # construct the argument parser and parse the arguments
# # command line arguments:
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True, help="path to output image containing ArUCo tag")
# ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
# ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
# args = vars(ap.parse_args())
#
# # verify that the supplied ArUCo tag exist and is supported by OpenCV
# if ARUCO_DICT.get(args["type"], None) is None:
#     print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
#     sys.exit(0)
#
# # Load the ArUCo dictionary
# arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
#
# # allocate memory for the output ArUCo tag and then draw the ArUCo tag on the output image
# print("[INFO] gererating ArUCo tag type '{}' with ID '{}'".format(args["type"], args["id"]))
# tag = np.zeros((300, 300, 1), dtype="uint8")
# cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, 1)
#
# # write the generated ArUCo tag to disk and then display it to our screen
# cv2.imwrite(args["output"], tag)
# cv2.imshow("ArUCo Tag", tag)
# cv2.waitKey(0)


def generate_marker(dictionary, id, side_pixels=420, border_bits=1, disp=True, save=False, path='markers'):
    """
    Creates & saves a canonical marker image.

    :parameter:
        dictionary (str/int): Dictionary of markers indicating the type of markers.
        id (int): Valid identifier of the marker that will be returned.
        side_pixels (int): Size of the image in pixels.
        border_bits (int): Width of the marker border.
        disp (bool): Flag specifying if marker will be displayed.
        save (bool): Flag specifying if marker will be saved on disc.
        path (str): Path to destination where markers will be saved.
    :return:
        img (numpy.ndarray): Canonical marker image.

    """
    # Verify that the supplied dict exist and is supported by OpenCV
    if ARUCO_DICT.get(dictionary, None) is None:
        raise ValueError("No such dictionary as '{}'".format(dictionary))

    # Load the ArUCo dictionary
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dictionary])

    # Verify that the supplied tag ID exist and is supported by OpenCV
    if id >= len(aruco_dict.bytesList):
        raise ValueError("Tag ID '{}' doesn't exist in {}".format(id, dictionary))

    # Allocate memory for the output ArUCo tag and then draw the ArUCo tag on the output image
    tag = np.zeros((side_pixels, side_pixels, 1), dtype="uint8")
    cv2.aruco.drawMarker(aruco_dict, id, side_pixels, tag, border_bits)

    # Write the generated ArUCo tag to disc
    if save:
        cv2.imwrite("{}\\Tag no.{} from {}.png".format(path, id, dictionary), tag)

    # Display the generated ArUCo tag on screen
    if disp:
        cv2.imshow("ArUCo Marker", tag)
        cv2.waitKey(0)

    return tag


if __name__ == '__main__':
    generate_marker('DICT_5X5_250', 10, save=True)

# TODO: Umożliwić wpisywanie argumentu 'dict' jako inta
