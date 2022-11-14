#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from constants import ARUCO_DICT
from typing import Tuple, Union, Optional


def detect_on_image(image: np.ndarray, dict_name: Optional[str] = None, disp: bool = False, show_rejected: bool = False, show_dict: bool = True, preview_resolution: Optional[Tuple[int, int]] = None) -> Tuple[Tuple, np.ndarray, Tuple]:
    """Detects & displays aruco marker on the given image.

    Args:
        image (array-like): Image to be analyzed.
        dict_name (str, optional): Indicates the type of markers that will be searched. Automatic detection if None.
        disp (bool, optional): Determines if the result image will be displayed.
        show_rejected (bool, optional): Specifies if rejected figures will be displayed on the result image.
        show_dict (bool, optional): Specifies if searched dict_name name will be displayed on the result image.
        preview_resolution (Tuple[int, int], optional): Resolution of the displayed image.

    Returns:
        A Tuple with 3 array-like vectors. First of them contains detected marker corners. For each marker, its four
        corners are provided. The second is vector of identifiers of the detected markers. The identifier is of
        type int. Third vector includes ImgPoints of those squares whose inner code has not a correct codification.

    Raises:
        ValueError: If given dict_name is not valid

    """
    if dict_name is None:
        max_spot_num = 0
        detection_results = ([], [], [])
        chosen_dict_name = ''
        display_text = 'Auto dict: '

        # Loop over the types of ArUco dictionaries
        for (dict_name, dict_enum) in ARUCO_DICT.items():
            # Attempt to detect the markers for the current dict
            aruco_dict = cv2.aruco.Dictionary_get(dict_enum)
            aruco_params = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

            # Check if number of detected markers was greater than previous record
            if len(corners) > max_spot_num:
                # Write results to temp variables
                max_spot_num = len(corners)
                detection_results = (corners, ids, rejected)
                chosen_dict_name = dict_name

        # Rewrite variables using final detection results
        corners, ids, rejected = detection_results
        dict_name = chosen_dict_name
    else:
        display_text = 'Manual dict: '

        # Verify that the supplied dict exist and is supported by OpenCV
        if ARUCO_DICT.get(dict_name, None) is None:
            raise ValueError("No such dict_name as '{}'".format(dict_name))

        # Attempt to detect the markers for the given dict
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
        aruco_params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    if disp:
        image = draw_markers_on_image(image, corners, ids)

        if show_rejected:
            image = draw_rejected_on_image(image, rejected)

        if show_dict:
            image = draw_dict_on_image(image, display_text, dict_name)

        if preview_resolution is not None and preview_resolution[0] > 0 and preview_resolution[1] > 0:
            image = cv2.resize(image, preview_resolution)

        # Show the output image
        cv2.imshow("Detection result", image)
        cv2.waitKey(0)

    return corners, ids, rejected


def detect_on_video(source: Union[str, int] = 0, dict_name: Optional[str] = None, show_rejected: bool = False, show_dict: bool = True, resolution: Optional[Tuple[int, int]] = None, preview_resolution: Optional[Tuple[int, int]] = None) -> None:
    """Detects & displays aruco marker on the given video or webcam.

    Args:
        source (str or int, optional): Path to video file or device index. If 0, primary camera (webcam) will be used.
        dict_name (str, optional): Indicates the type of markers that will be searched. Automatic detection if None.
        show_rejected (bool, optional): Specifies if rejected figures will be displayed on the result video.
        show_dict (bool, optional): Specifies if searched dict_name name will be displayed on the result video.
        resolution (Tuple[int, int], optional): Resolution of the captured video.
        preview_resolution (Tuple[int, int], optional): Resolution of the displayed video.

    Raises:
        ValueError: If given dict_name is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    if dict_name is None:
        display_text = 'Auto dict: '
        aruco_params = cv2.aruco.DetectorParameters_create()
        cv2.namedWindow("Preview")

        # Check data type of source argument
        if isinstance(source, int) or isinstance(source, str):
            vc = cv2.VideoCapture(source)
            # Check whether the camera resolution is to be changed
            if resolution is not None and resolution[0] > 0 and resolution[1] > 0:
                vc.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                vc.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
                height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if resolution[0] != width_set or resolution[1] != height_set:
                    print("Specified camera resolution could not be set.")
                    print(f"{int(width_set)}x{int(height_set)} resolution is currently used.")
                else:
                    print(f"Using the specified camera resolution {int(width_set)}x{int(height_set)}.")
            else:
                width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
                height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Using the default camera resolution {int(width_set)}x{int(height_set)}.")
            # Preview resolution info log
            if preview_resolution is not None and preview_resolution[0] > 0 and preview_resolution[1] > 0:
                print(f"The preview resolution is set to {int(preview_resolution[0])}x{int(preview_resolution[1])}.")
            else:
                print("The preview resolution is the same as camera resolution.")
        else:
            raise TypeError("Source parameter does not accept {}".format(type(source)))

        # Try to get the first frame
        if (vc is not None) and vc.isOpened():
            rval, frame = vc.read()
        else:
            raise SystemError("Unable to open video source: {}".format(source))

        # Loop until there are no frames left
        while rval:
            max_spot_num = 0
            detection_results = ([], [], [])
            chosen_dict_name = ''

            # Loop over the types of ArUco dictionaries
            for (dict_name, dict_enum) in ARUCO_DICT.items():
                # Attempt to detect the markers for the current dict
                aruco_dict = cv2.aruco.Dictionary_get(dict_enum)
                corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

                # Check if number of detected markers was greater than previous record
                if len(corners) > max_spot_num:
                    max_spot_num = len(corners)
                    detection_results = (corners, ids, rejected)
                    chosen_dict_name = dict_name

            # Rewrite variables using final detection results
            corners, ids, rejected = detection_results
            dict_name = chosen_dict_name
            frame = draw_markers_on_image(frame, corners, ids)

            if show_rejected:
                frame = draw_rejected_on_image(frame, rejected)

            if show_dict:
                frame = draw_dict_on_image(frame, display_text, dict_name)

            if preview_resolution is not None and preview_resolution[0] > 0 and preview_resolution[1] > 0:
                frame = cv2.resize(frame, preview_resolution)

            # Update the output image
            cv2.imshow("Preview", frame)
            rval, frame = vc.read()

            key = cv2.waitKey(20)
            # Exit if ESC key button or X window button pressed
            if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                break
    else:
        # Verify that the supplied dict exist and is supported by OpenCV
        if ARUCO_DICT.get(dict_name, None) is None:
            raise ValueError("No such dict_name as '{}'".format(dict_name))

        display_text = 'Manual dict: '
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
        aruco_params = cv2.aruco.DetectorParameters_create()
        cv2.namedWindow("Preview")

        # Check data type of source argument
        if isinstance(source, int) or isinstance(source, str):
            vc = cv2.VideoCapture(source)
            # Check whether the camera resolution is to be changed
            if resolution is not None and resolution[0] > 0 and resolution[1] > 0:
                vc.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                vc.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
                height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if resolution[0] != width_set or resolution[1] != height_set:
                    print("Specified camera resolution could not be set.")
                    print(f"{int(width_set)}x{int(height_set)} resolution is currently used.")
                else:
                    print(f"Using the specified camera resolution {int(width_set)}x{int(height_set)}.")
            else:
                width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
                height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"Using the default camera resolution {int(width_set)}x{int(height_set)}.")
            # Preview resolution info log
            if preview_resolution is not None and preview_resolution[0] > 0 and preview_resolution[1] > 0:
                print(f"The preview resolution is set to {int(preview_resolution[0])}x{int(preview_resolution[1])}.")
            else:
                print("The preview resolution is the same as camera resolution.")
        else:
            raise TypeError("Source parameter does not accept {}".format(type(source)))

        # Try to get the first frame
        if (vc is not None) and vc.isOpened():
            rval, frame = vc.read()
        else:
            raise SystemError("Unable to open video source: {}".format(source))

        # Loop until there are no frames left
        while rval:
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
            frame = draw_markers_on_image(frame, corners, ids)

            if show_rejected:
                frame = draw_rejected_on_image(frame, rejected)

            if show_dict:
                frame = draw_dict_on_image(frame, display_text, dict_name)

            if preview_resolution is not None and preview_resolution[0] > 0 and preview_resolution[1] > 0:
                frame = cv2.resize(frame, preview_resolution)

            # Update the output image
            cv2.imshow("Preview", frame)
            rval, frame = vc.read()

            key = cv2.waitKey(10)
            # Exit if ESC key button or X window button pressed
            if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
                break

    vc.release()
    cv2.destroyAllWindows()


def draw_markers_on_image(image: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Draw detected markers and their ids on the image.

    Args:
        image (array-like): Image where shapes will be drawn.
        corners (array-like): Corners of markers.
        ids (array-like): Ids of markers.

    Returns:
        array-like: Image with markers and ids drawn on.

    """
    # Verify if at last one ArUCo marker was detected
    if len(corners) > 0:
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

    return image


def draw_rejected_on_image(image: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    """Draw rejected figures on the image.

    Args:
        image (array-like): Image where shapes will be drawn.
        rejected (array-like): Rejected figures.

    Returns:
        array-like: Image with rejected figures drawn on.

    """
    # Loop over the rejected ArUCo corners
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

    return image


def draw_dict_on_image(image: np.ndarray, det_type: str, dict_name: str) -> np.ndarray:
    """Draw searched dict name on the image.

    Args:
        image (array-like): Image where shapes will be drawn.
        det_type (str): Type of detection.
        dict_name (str): Detected dict.

    Returns:
        array-like: Image with dict name drawn on.

    """
    display_text = det_type + str(dict_name)
    cv2.putText(image, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
    return image


if __name__ == '__main__':
    # Load the input image fom disk
    # path = 'images\\test_images\\test5.jpg'
    # dict = "DICT_4X4_50"
    # image = cv2.imread(path)

    #detect_on_image(image, disp=True, show_rejected=False, show_dict=True, preview_resolution=(1400, 700))
    #detect_on_video("C:\\Users\\micha\\Pulpit\\Życie prywatne\\Filmy\\Drift1.mp4", "DICT_4X4_50", show_rejected=True)
    detect_on_video(0, resolution=(1400, 700))

# TODO: Dodoać argparsera z możliwością wyboru danej funkcji
