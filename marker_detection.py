#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import sys
from constants import ARUCO_DICT


def detect_on_image(image, dict_name=None, disp=True, show_rejected=True, show_dict=False, resize=True):
    if dict_name is None:
        # loop over the types of ArUco dictionaries
        max_spot_num = 0
        detection_results = ([], [], [])
        chosen_dict_name = ''
        display_text = 'Auto dict: '
        for (dict_name, dict_enum) in ARUCO_DICT.items():
            # load the ArUCo dict_name, grab the ArUCo parameters, and attempt to detect the markers for the current dict_name
            aruco_dict = cv2.aruco.Dictionary_get(dict_enum)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
            if len(corners) > max_spot_num or (max_spot_num == 0 and dict_enum == 21):
                max_spot_num = len(corners)
                detection_results = (corners, ids, rejected)
                if max_spot_num != 0:
                    chosen_dict_name = dict_name
            print("[INFO] detected {} markers for '{}'".format(len(corners), dict_name))
        corners, ids, rejected = detection_results
        dict_name = chosen_dict_name
    else:
        display_text = 'Manual dict: '
        # Verify that the supplied dict exist and is supported by OpenCV
        if ARUCO_DICT.get(dict_name, None) is None:
            raise ValueError("No such dict_name as '{}'".format(dict_name))
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    # Verify if at last one ArUCo marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        print("[INFO] detected {} markers for '{}'".format(len(corners), dict_name))
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

    if show_dict:
        # Draw the ArUco markers dict on the image
        display_text = display_text + str(dict_name)
        cv2.putText(image, display_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    return None


def detect_on_video(dict_name=None):
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

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
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Compute and draw the center coordinates of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the image
                cv2.putText(frame, str(marker_id), (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")

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

    # Load the input image from disk
    path = 'real_images//test11.jpg'
    dict = "DICT_4X4_50"
    image = cv2.imread(path)

    detect_on_image(image, disp=True, show_rejected=False, resize=True, show_dict=True)
    #detect_on_video("DICT_4X4_50")

# TODO:
