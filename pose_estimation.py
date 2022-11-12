#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

from camera_calibration import load_coefficients
from marker_detection import detect_on_image
from typing import Optional, Union, Tuple


def is_rotation_matrix(rot_mtx: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix.

    Args:
        rot_mtx (array-like): Matrix to be checked.

    Returns:
        bool: Test result

    """
    should_be_identity = np.dot(np.transpose(rot_mtx), rot_mtx)
    should_be_one = np.linalg.det(rot_mtx)
    is_identity = np.allclose(should_be_identity, np.identity(rot_mtx.shape[0], dtype=rot_mtx.dtype))
    is_one = np.allclose(should_be_one, 1.0)
    return is_identity and is_one


def rotation_matrix_to_euler_angles(rot_mtx: np.ndarray) -> np.array:
    """Calculates euler angles from rotation matrix.

    Args:
        rot_mtx (array-like): Valid 3x3 rotation matrix.

    Returns:
        np.array: Rotation around x, y and z axes in radians.

    """
    if not is_rotation_matrix(rot_mtx):
        raise ValueError("Object is not a rotation matrix")

    sy = math.sqrt(rot_mtx[0, 0] * rot_mtx[0, 0] + rot_mtx[1, 0] * rot_mtx[1, 0])
    is_singular = sy < 1e-6

    if not is_singular:
        x_rot = math.atan2(rot_mtx[2, 1], rot_mtx[2, 2])
        y_rot = math.atan2(-rot_mtx[2, 0], sy)
        z_rot = math.atan2(rot_mtx[1, 0], rot_mtx[0, 0])
    else:
        x_rot = math.atan2(-rot_mtx[1, 2], rot_mtx[1, 1])
        y_rot = math.atan2(-rot_mtx[2, 0], sy)
        z_rot = 0

    return np.array([x_rot, y_rot, z_rot])


def estimate_markers_pose_on_image(image: np.ndarray, marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, dict_name: Optional[str] = None, disp: bool = True, max_dim: Optional[Tuple[int, int]] = None) -> Tuple[Tuple, np.array, np.array, np.array]:
    """Estimates the pose of each individual marker on the image."""
    # Detect aruco markers
    corners_list, ids, r = detect_on_image(image=image, dict_name=dict_name, disp=False)

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

    return corners_list, ids, np.array(rvec_list), np.array(tvec_list)


def estimate_markers_pose_on_video(marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, source: Union[str, int] = 0, dict_name: Optional[str] = None, max_dim: Optional[Tuple[int, int]] = None, resolution: Optional[Tuple[int, int]] = None) -> None:
    """Estimates the camera posture using single ArUco marker on the image."""
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
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Loop until there are no frames left
    while rval:
        # Detect aruco markers
        corners_list, _, _ = detect_on_image(image=frame, dict_name=dict_name, disp=False)

        # Loop over every detected marker's corners
        for corners in corners_list:
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, mtx, dist)
            rvec = rvec[0][0]
            tvec = tvec[0][0]
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, [corners])
            # Draw axis of the marker
            cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, marker_len/2)


            # Obtain the rotation matrix tag->camera
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotation_matrix_t = rotation_matrix.T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix_t)
            # realworld_tvec
            # pos_cam = np.dot(rotation_matrix, tvec_flipped)
            # pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
            # Get position and attitude of rhe camera respt to the marker
            rot_cam = [roll, pitch, yaw]
            display_text2 = f"X = {int(tvec[0])}   Y = {int(tvec[1])}   Z = {int(tvec[2])}"
            display_text3 = f"R = {int(math.degrees(rot_cam[0]))}   P = {int(math.degrees(rot_cam[1]))}   Y = {int(math.degrees(rot_cam[2]))}"

            print(display_text2)
            print(display_text3)

            cv2.putText(frame, "Marker position:", (500, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text2, (500, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, "Marker rotation:", (500, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text3, (500, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

            pos_cam = np.matrix(tvec).T
            #pos_cam = -rotation_matrix_t*np.matrix(tvec).T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix_t)
            # Get position and attitude of rhe camera respt to the marker
            rot_cam = [roll, pitch, yaw]
            display_text2 = f"X = {int(pos_cam[0])}   Y = {int(pos_cam[1])}   Z = {int(pos_cam[2])}"
            display_text3 = f"R = {int(math.degrees(rot_cam[0]))}   P = {int(math.degrees(rot_cam[1]))}   Y = {int(math.degrees(rot_cam[2]))}"

            print(display_text2)
            print(display_text3)

            cv2.putText(frame, "Camera position2:", (5, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text2, (5, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, "Camera rotation2:", (5, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text3, (5, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        if max_dim is not None:
            # Max image dimensions
            max_width, max_height = max_dim

            # Current dimensions
            height, width, _ = np.shape(frame)

            # Check if any of the image dim is bigger than max dim
            if width > max_width or height > max_height:
                new_dim = (max_height, max_width)
                frame = cv2.resize(frame, new_dim)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


def estimate_camera_pose_on_image(image: np.ndarray, marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, dict_name: Optional[str] = None, disp: bool = True, max_dim: Optional[Tuple[int, int]] = None):
    """Estimates the camera posture using single ArUco marker on the image."""
    # Detect aruco markers
    corners_list, _, _ = detect_on_image(image=image, dict_name=dict_name, disp=False)

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
        roll_camera, pitch_camera, yaw_camera = rotation_matrix_to_euler_angles(R_flip*R_tc)
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


def estimate_camera_pose_on_video(marker_len: Union[int, float], mtx: np.ndarray, dist: np.array, source: Union[str, int] = 0, dict_name: Optional[str] = None, max_dim: Optional[Tuple[int, int]] = None, resolution: Optional[Tuple[int, int]] = None):
    """Estimates the camera posture using single ArUco marker on the image."""
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
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Loop until there are no frames left
    while rval:
        # Detect aruco markers
        corners_list, _, _ = detect_on_image(image=frame, dict_name=dict_name, disp=False)

        num_markers = len(corners_list)
        # Check how many markers were detected
        if num_markers == 1:
            display_text1 = "Data based on the detected marker"

            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_list[0], marker_len, mtx, dist)
            rvec = rvec[0][0]
            tvec = tvec[0][0]

            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners_list)
            # Draw axis of the marker
            cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, marker_len/2)

            rvec_flipped = rvec * -1
            tvec_flipped = tvec * -1
            # Obtain the rotation matrix tag->camera
            rotation_matrix, jacobian = cv2.Rodrigues(rvec_flipped)
            # realworld_tvec
            pos_cam = np.dot(rotation_matrix, tvec_flipped)
            pitch, roll, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
            # Get position and attitude of rhe camera respt to the marker
            rot_cam = [roll, pitch, yaw]
            display_text2 = f"X = {int(pos_cam[0])}   Y = {int(pos_cam[1])}   Z = {int(pos_cam[2])}"
            display_text3 = f"R = {int(math.degrees(rot_cam[0]))}   P = {int(math.degrees(rot_cam[1]))}   Y = {int(math.degrees(rot_cam[2]))}"
        elif num_markers > 1:
            display_text1 = "More than one marker detected"
            display_text2 = "X =   Y =   Z = "
            display_text3 = f"R =   P =   Y = "
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners_list)
        else:
            display_text1 = "No marker detected"
            display_text2 = "X =   Y =   Z = "
            display_text3 = f"R =   P =   Y = "

        cv2.putText(frame, display_text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
        cv2.putText(frame, "Camera position:", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
        cv2.putText(frame, display_text2, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
        cv2.putText(frame, "Camera rotation:", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
        cv2.putText(frame, display_text3, (5, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        if max_dim is not None:
            # Max image dimensions
            max_width, max_height = max_dim

            # Current dimensions
            height, width, _ = np.shape(frame)

            # Check if any of the image dim is bigger than max dim
            if width > max_width or height > max_height:
                new_dim = (max_height, max_width)
                frame = cv2.resize(frame, new_dim)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(20)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    mtx, dist = load_coefficients('calibration_aruco_1280x720.yml')

    import pathlib
    path = r'C:\Users\micha\Pulpit\Test_aruco'
    img_dir = pathlib.Path(path)
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*.jpg'):  # Tu trzeba dać sprawdzanie czy ta lokalizacja nie będzie pusta
        image = cv2.imread(str(img))
        #estimate_markers_pose_on_image(image, 100, mtx, dist)
    estimate_markers_pose_on_image(cv2.imread(r'C:\Users\micha\Pulpit\Test_aruco\WIN_20221111_20_03_22_Pro.jpg'), 100, mtx, dist)
    estimate_markers_pose_on_image(cv2.imread(r'C:\Users\micha\Pulpit\Test_aruco\WIN_20221111_19_41_15_Pro.jpg'), 100, mtx, dist)

    estimate_markers_pose_on_video(100, mtx, dist, 0, resolution=(1280, 720))


    #estimate_camera_pose_on_video(100, mtx, dist, 0, resolution=(1280, 720))
    #estimate_camera_pose_on_video(26.5, mtx, dist)
    # image = cv2.imread('images\\calibration_images\\1\\aruco_calib_20.jpg')
    # print(np.shape(image))
    # print(estimate_markers_pose_on_image(image, 26.5, mtx, dist, dict_name=None, disp=True))
    # image = cv2.imread('images\\test_images\\camera_pos_test_3.jpg')
    # estimate_camera_pose_on_image(image, 26.5, mtx, dist, dict_name=None, disp=True)

# TODO: Upewnić się co ma zwracać funkcja 'estimate_pose_on_image()'
# TOdO: Sprawdzć czas miedzy obrazem w gray scale a kolorowym
