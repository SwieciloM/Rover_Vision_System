#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

from camera_calibration import load_coefficients
from marker_detection import detect_on_image
from typing import Optional, Union, Tuple


def estimate_markers_pose_on_image(image: np.ndarray, marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, dict_name: Optional[str] = None, disp: bool = True, max_dim: Optional[Tuple[int, int]] = None):
    """Estimates the pose of each individual marker on the image."""
    # Detect aruco markers
    corners_list, _, _ = detect_on_image(image=image, dict_name=dict_name, disp=False)

    rvec_list, tvec_list = [], []
    # Check if markers were detected
    if len(corners_list) > 0:
        # Loop over every detected marker's corners
        for corners in corners_list:
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, mtx, dist)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(image, [corners])
            # Draw axis of the marker
            cv2.aruco.drawAxis(image, mtx, dist, rvec, tvec, marker_len/2)
            rvec_list.append(rvec)
            tvec_list.append(tvec)

    if disp:
        if max_dim is not None:
            # Max image dimensions
            max_width, max_height = max_dim

            # Current dimensions
            height, width, _ = np.shape(image)

            # Check if any of the image dim is bigger than max dim
            if width > max_width or height > max_height:
                new_dim = (max_height, max_width)
                image = cv2.resize(image, new_dim)

        # Show the output image
        cv2.imshow("Detection result", image)
        cv2.waitKey(0)

    return corners_list, rvec_list, tvec_list


def estimate_pose_on_video(source: Union[str, int] = 0, dict_name: Optional[str] = None, disp: bool = True, show_rejected: bool = False, show_dict: bool = True) -> None:
    pass


def estimate_camera_pose_on_image(image: np.ndarray, marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, dict_name: Optional[str] = None, disp: bool = True, max_dim: Optional[Tuple[int, int]] = None):
    """Estimates the camera posture using single ArUco marker on the image."""
    # Detect aruco markers
    corners_list, _, _ = detect_on_image(image=image, dict_name=dict_name, disp=False)

    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(R):
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    num_markers = len(corners_list)
    # Check how many markers were detected
    if num_markers == 1:
        R_flip = np.zeros((3, 3), dtype=np.float32)
        R_flip[0, 0] = 1.0
        R_flip[1, 1] = -1.0
        R_flip[2, 2] = -1.0


        # Estimate pose of each marker and return the values rvec and tvec
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_list[0], marker_len, mtx, dist)
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(image, corners_list)
        # Draw axis of the marker
        cv2.aruco.drawAxis(image, mtx, dist, rvec, tvec, marker_len/2)
        # Obtain the rotation matrix tag->camera
        R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
        R_tc = R_ct.T
        # Get position and attitude of rhe camera respt to the marker
        pos_camera = -R_tc*np.matrix(tvec).T
        roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
        # Display actual number of saved images
        display_text = f"Camera position:\n X = {pos_camera[0]}   Y = {pos_camera[1]}   Z = {pos_camera[2]}"
        cv2.putText(image, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
        display_text = f"Camera rotation:\n R = {roll_camera}   P = {pitch_camera}   Y = {yaw_camera}"
        cv2.putText(image, display_text, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
    elif num_markers > 1:
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(image, corners_list)

    else:
        pass

    if disp:
        if max_dim is not None:
            # Max image dimensions
            max_width, max_height = max_dim

            # Current dimensions
            height, width, _ = np.shape(image)

            # Check if any of the image dim is bigger than max dim
            if width > max_width or height > max_height:
                new_dim = (max_height, max_width)
                image = cv2.resize(image, new_dim)

        # Show the output image
        cv2.imshow("Detection result", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # cv2.namedWindow("Preview")
    # vc = cv2.VideoCapture(0)
    # mtx, dist = load_coefficients('calibration_chess.yml')
    #
    # if vc.isOpened():
    #     rval, frame = vc.read()
    #
    # # Loop until there are no frames left
    # while rval:
    #     frame = estimate_pose_on_image(frame, 26.5, mtx, dist, dict_name=None, disp=False)
    #     # Update the output image
    #     cv2.imshow("Preview", frame)
    #     rval, frame = vc.read()
    #
    #     key = cv2.waitKey(10)
    #     # Exit if ESC key button or X window button pressed
    #     if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
    #         break
    #
    # vc.release()
    # cv2.destroyAllWindows()

    mtx, dist = load_coefficients('calibration_chess.yml')
    image = cv2.imread('images\\calibration_images\\1\\aruco_calib_20.jpg')
    print(np.shape(image))
    estimate_markers_pose_on_image(image, 26.5, mtx, dist, dict_name=None, disp=True)
    image = cv2.imread('images\\test_images\\camera_pos_test_3.jpg')
    estimate_camera_pose_on_image(image, 26.5, mtx, dist, dict_name=None, disp=True)

    # Load coefficients
    mtx, dist = load_coefficients('calibration_chess.yml')
    original = cv2.imread('images\\calibration_images\\2\\chess_calib_20.jpg')
    cv2.imshow('Oryginalne zdjecie chess', original)
    undst = cv2.undistort(original, mtx, dist, None, mtx)
    cv2.imshow('Odnieksztalcone zdjecie chess', undst)

    mtx, dist = load_coefficients('calibration_charuco.yml')
    #original = cv2.imread('images\\calibration_images\\3\\charuco_calib_20.jpg')
    cv2.imshow('Oryginalne zdjecie charuco', original)
    undst = cv2.undistort(original, mtx, dist, None, mtx)
    cv2.imshow('Odnieksztalcone zdjecie charuco', undst)

    mtx, dist = load_coefficients('calibration_aruco.yml')
    #original = cv2.imread('images\\calibration_images\\1\\aruco_calib_20.jpg')
    cv2.imshow('Oryginalne zdjecie aruco', original)
    undst = cv2.undistort(original, mtx, dist, None, mtx)
    cv2.imshow('Odnieksztalcone zdjecie aruco', undst)

    cv2.waitKey(0)
    # cv2.imwrite('images\\calibration_images\\undist_chess.png', dst)


# TODO: Upewnić się co ma zwracać funkcja 'estimate_pose_on_image()'