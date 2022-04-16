#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import sys
from constants import ARUCO_DICT


def detect_on_image(image, dictionary=None, disp=True, show_rejected=True, resize=True):
    if dictionary is None:
        # Tutaj będzie algorytm auto rozpoznawania słowników
        pass
    else:
        # Verify that the supplied dict exist and is supported by OpenCV
        if ARUCO_DICT.get(dictionary, None) is None:
            raise ValueError("No such dictionary as '{}'".format(dictionary))
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dictionary])

    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    # Verify if at last one ArUCo marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

    # Loop over the detected ArUCo corners
    for (marker_corners, marker_id) in zip(corners, ids):
        # Extract the marker corners (top-left, top-right, bottom-right and bottom-left order)
        corners = marker_corners.reshape((4, 2))
        # Convert each of the (x,y) coordinate pairs to integers
        top_left = (int(corners[0][0]), int(corners[0][1]))
        top_right = (int(corners[1][0]), int(corners[1][1]))
        bottom_right = (int(corners[2][0]), int(corners[2][1]))
        bottom_left = (int(corners[3][0]), int(corners[3][1]))

        # Draw the bounding box of the ArUCo detection
        cv2.line(image, top_left, top_right, (0, 255, 0), 2)
        cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)

        # Compute and draw the center coordinates of the ArUco marker
        center_x = int((top_left[0] + bottom_right[0])/2)
        center_y = int((top_left[1] + bottom_right[1])/2)
        cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)

        # Draw the ArUco marker ID on the image
        cv2.putText(image, str(marker_id), (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] ArUco marker ID: {}".format(marker_id))

    if show_rejected:
        # loop over the detected ArUCo corners
        for fig_corners in rejected:
            # Extract the rejected marker corners (top-left, top-right, bottom-right and bottom-left order)
            corners = fig_corners.reshape((4, 2))
            # Convert each of the (x,y) coordinate pairs to integers
            top_left = (int(corners[0][0]), int(corners[0][1]))
            top_right = (int(corners[1][0]), int(corners[1][1]))
            bottom_right = (int(corners[2][0]), int(corners[2][1]))
            bottom_left = (int(corners[3][0]), int(corners[3][1]))

            # Draw the bounding box of the ArUCo detection
            cv2.line(image, top_left, top_right, (0, 0, 255), 2)
            cv2.line(image, top_right, bottom_right, (0, 0, 255), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 0, 255), 2)
            cv2.line(image, bottom_left, top_left, (0, 0, 255), 2)

            # Compute and draw the center coordinates of the ArUco marker
            # center_x = int((top_left[0] + bottom_right[0]) / 2)
            # center_y = int((top_left[1] + bottom_right[1]) / 2)
            # cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)

    if resize:
        img_shape = np.shape(image)
        if img_shape[0] > 600:
            k = 600/img_shape[0]
        elif img_shape[1] > 1000:
            k = 1000/img_shape[1]
        else:
            k = 1
        dim = (int(img_shape[1] * k), int(img_shape[0] * k))
        r_image = cv2.resize(image, dim)

        # Show the output image
        cv2.imshow("Detection result", r_image)
        cv2.waitKey(0)
    else:
        # Show the output image
        cv2.imshow("Detection result", image)
        cv2.waitKey(0)

    return ids


def detect_on_video():
    pass


if __name__ == '__main__':
    # # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to input image containing ArUCo tag")
    # ap.add_argument("-t", "--type", type=str,
    #                 default="DICT_ARUCO_ORIGINAL",
    #                 help="type of ArUCo tag to detect")
    # args = vars(ap.parse_args())

    # load the input image from disk and resize it
    print("[INFO] loading image...")
    path = 'real_images//test5.jpg'
    dict = "DICT_7X7_50"
    image = cv2.imread(path)
    #image = imutils.resize(image, width=600)

    detect_on_image(image, "DICT_7X7_50", disp=True, show_rejected=False, resize=True)
