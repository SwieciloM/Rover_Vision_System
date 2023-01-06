#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import degrees, sqrt, atan2
from typing import Optional, Union, Tuple
from camera_calibration import load_coefficients
from marker_detection import detect_on_image


def initialize_videocapture(source: Union[str, int], src_res: Optional[Tuple[int, int]] = None) -> cv2.VideoCapture:
    """Gets the VideoCapture object.

    Args:
        source (str or int): Path to video file or device index.
        src_res (Tuple[int, int], optional): Resolution of the captured video.

    Returns:
        cv2.VideoCapture: Defined VideoCapture object.

    Raises:
        TypeError: If given source argument has wrong type.

    """
    # Check data type of source argument
    if isinstance(source, int) or isinstance(source, str):
        vc = cv2.VideoCapture(source)

        # Check whether the camera resolution is to be changed
        if src_res is not None and src_res[0] > 0 and src_res[1] > 0:
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, src_res[0])
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, src_res[1])
            width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
            height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if src_res[0] != width_set or src_res[1] != height_set:
                print("Specified camera resolution could not be set.")
                print(f"{int(width_set)}x{int(height_set)} resolution is currently used.")
            else:
                print(f"Using the specified camera resolution {int(width_set)}x{int(height_set)}.")

        else:
            width_set = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
            height_set = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Using the default camera resolution {int(width_set)}x{int(height_set)}.")

        return vc

    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))


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

    sy = sqrt(rot_mtx[0, 0] * rot_mtx[0, 0] + rot_mtx[1, 0] * rot_mtx[1, 0])
    is_singular = sy < 1e-6

    if not is_singular:
        x_rot = atan2(rot_mtx[2, 1], rot_mtx[2, 2])
        y_rot = atan2(-rot_mtx[2, 0], sy)
        z_rot = atan2(rot_mtx[1, 0], rot_mtx[0, 0])
    else:
        x_rot = atan2(-rot_mtx[1, 2], rot_mtx[1, 1])
        y_rot = atan2(-rot_mtx[2, 0], sy)
        z_rot = 0

    return np.array([x_rot, y_rot, z_rot])


def estimate_markers_pose_on_image(image: np.ndarray, marker_len: Union[int, float], cam_mtx: np.ndarray, dist_coefs: np.ndarray, dict_name: Optional[str] = None, disp: bool = False, show_values: bool = True, show_ids: bool = False, show_axis: bool = True, ret_final: bool = False, prev_res: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple, np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the position of each individual marker on the image.

    Markers must be of the same size and type in order to assess their position correctly.

    Args:
        image (np.ndarray): Image to be analyzed.
        marker_len (int or float): Side length of the marker in millimeters.
        cam_mtx (np.ndarray): Camera matrix.
        dist_coefs (np.ndarray): Distortion coefficients.
        dict_name (str, optional): Type of ArUco marker. Passing it may speed up the program.
        disp (bool, optional): Determines if the result image will be displayed.
        show_values (bool, optional): When it is True, the translation [cm] and rotation [deg] values are displayed.
        show_ids (bool, optional): When it is True, marker's ids are displayed.
        show_axis (bool, optional): When it is True, marker's axis are displayed.
        ret_final (bool, optional): When it is True, the final image is returned.
        prev_res (Tuple[int, int], optional): Resolution of the displayed image.

    Returns:
        Tuple: Image, Corners, IDs, Rotation and Translation vectors of each marker.

    """
    # Detect aruco markers
    corners_list, ids, _ = detect_on_image(image=image, dict_name=dict_name, disp=False)

    rvec_list, tvec_list = [], []

    # Check if there are any detected markers
    if len(corners_list):
        # Loop over every detected marker's corners
        for corners in corners_list:
            # Estimate pose of the marker to obtain rotation and translation vectors
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, cam_mtx, dist_coefs)
            rvec_list.append(rvec)
            tvec_list.append(tvec)

    # Prepare the final image
    if disp or ret_final:
        # Check if there are any detected markers
        if len(corners_list):
            # Loop over every detected marker's data
            for corners, id, rvec, tvec in zip(corners_list, ids, rvec_list, tvec_list):
                rvec = rvec[0][0]
                tvec = tvec[0][0]

                # Draw a square around detected marker
                if show_ids:
                    cv2.aruco.drawDetectedMarkers(image, [corners], id)
                else:
                    cv2.aruco.drawDetectedMarkers(image, [corners])

                # Draw axis of the marker
                if show_axis:
                    cv2.aruco.drawAxis(image, cam_mtx, dist_coefs, rvec, tvec, marker_len/2)

                if show_values:
                    # Obtain the rotation matrix to get euler angles
                    rot_mtx_t = cv2.Rodrigues(rvec)[0].T
                    roll, pitch, yaw = rotation_matrix_to_euler_angles(rot_mtx_t)

                    # Text to display
                    tra_text = "({:.0f}, {:.0f}, {:.0f})".format(tvec[0]/10, tvec[1]/10, tvec[2]/10)
                    rot_text = "({:.0f}, {:.0f}, {:.0f})".format(degrees(roll), degrees(pitch), degrees(yaw))

                    # Parameters for correct text display
                    x_txt, y_txt = [int(min(i)) for i in zip(*corners[0])]
                    size_marker = [int(max(i) - min(i)) for i in zip(*corners[0])]
                    font_scale = sum(size_marker)/450
                    offset = int(size_marker[1]/10)
                    color = (19, 111, 216)

                    # Draw rotation and translation values
                    cv2.putText(image, tra_text, (x_txt, y_txt+4*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                    cv2.putText(image, rot_text, (x_txt, y_txt+7*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

        # Resize final image
        if prev_res is not None and prev_res[0] > 0 and prev_res[1] > 0:
            image = cv2.resize(image, prev_res)

    if disp:
        # Display the final image
        cv2.imshow("Estimation result", image)
        cv2.waitKey(0)

    return image, corners_list, ids, np.array(rvec_list), np.array(tvec_list)


def estimate_markers_pose_on_video(source: Union[str, int], marker_len: Union[int, float], cam_mtx: np.ndarray, dist_coefs: np.ndarray, dict_name: Optional[str] = None, show_values: bool = True, show_ids: bool = False, show_axis: bool = True, src_res: Optional[Tuple[int, int]] = None, prev_res: Optional[Tuple[int, int]] = None) -> None:
    """Estimates the position of each individual marker on the video.

    Markers must be of the same size and type in order to assess their position correctly.

    Args:
        source (str or int): Path to video file or device index. Device id '0' is for built-in camera.
        marker_len (int or float): Side length of the marker in millimeter.
        cam_mtx (np.ndarray): Camera matrix.
        dist_coefs (np.ndarray): Distortion coefficients.
        dict_name (str, optional): Type of ArUco marker. Passing it may speed up the program.
        show_values (bool, optional): When it is True, the translation [cm] and rotation [deg] values are displayed.
        show_ids (bool, optional): When it is True, marker's ids are displayed.
        show_axis (bool, optional): When it is True, marker's axis are displayed.
        src_res (Tuple[int, int], optional): Resolution of the captured video.
        prev_res (Tuple[int, int], optional): Resolution of the displayed video.

    Raises:
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    vc = initialize_videocapture(source, src_res)

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Loop until there are no frames left
    while rval:
        # Get the frame with markers data on it
        frame, *_ = estimate_markers_pose_on_image(frame, marker_len, cam_mtx, dist_coefs, dict_name, False, show_values, show_ids, show_axis, True, prev_res)

        # Update the output image and get new frame
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        # Exit if ESC key button or X window button pressed
        key = cv2.waitKey(20)
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


def estimate_camera_pose_on_image(image: np.ndarray, marker_len: Union[int, float], cam_mtx: np.ndarray, dist_coefs: np.ndarray, dict_name: Optional[str] = None, disp: bool = False, show_values: bool = True, show_id: bool = False, show_axis: bool = True, ret_final: bool = False, prev_res: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the camera position using single ArUco marker on the image.

    Args:
        image (np.ndarray): Image to be analyzed.
        marker_len (int or float): Side length of the marker in millimeters.
        cam_mtx (np.ndarray): Camera matrix.
        dist_coefs (np.ndarray): Distortion coefficients.
        dict_name (str, optional): Type of ArUco marker. Passing it may speed up the program.
        disp (bool, optional): Determines if the result image will be displayed.
        show_values (bool, optional): When it is True, the translation [cm] and rotation [deg] values are displayed.
        show_id (bool, optional): When it is True, marker id is displayed.
        show_axis (bool, optional): When it is True, marker's axis are displayed.
        ret_final (bool, optional): When it is True, the final image is returned.
        prev_res (Tuple[int, int], optional): Resolution of the displayed image.

    Returns:
        Tuple: Image, Rotation and Translation vector of camera position.

    """
    # Detect aruco markers
    corners_list, ids, _ = detect_on_image(image=image, dict_name=dict_name, disp=False)

    rvec, tvec = [], []

    # Check how many markers were detected
    num_markers_detected = len(corners_list)
    if num_markers_detected:
        # Estimate pose of the first detected marker to obtain rotation and translation vector
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners_list[0], marker_len, mtx, dist)
        
        # Prepare the final image
        if disp or ret_final:
            rvec = rvec[0][0]
            tvec = tvec[0][0]

            # Draw a square around the first detected marker
            if show_id:
                cv2.aruco.drawDetectedMarkers(image, [corners_list[0]], ids[0])
            else:
                cv2.aruco.drawDetectedMarkers(image, [corners_list[0]])

            # Draw axis of the first detected marker
            if show_axis:
                cv2.aruco.drawAxis(image, cam_mtx, dist_coefs, rvec, tvec, marker_len/2)

            if show_values:
                # Obtain the inverse (in this case == transpose) of rotation matrix
                rot_mtx_t = np.transpose(cv2.Rodrigues(rvec)[0])
                # Get euler angles respect to the marker
                rcam = rotation_matrix_to_euler_angles(rot_mtx_t)
                # Get camera position respect to the marker
                pcam = -rot_mtx_t * np.matrix(tvec).T

                # Text to display
                if num_markers_detected > 1:
                    disp_text1 = " - Many markers detected. Estimation based on marker {} - ".format(ids[0])
                else:
                    disp_text1 = " - Estimation based on detected marker - "
                disp_text2 = "X = {} Y = {} Z = {}".format(int(pcam[0]/10), int(pcam[1]/10), int(pcam[2]/10))
                disp_text3 = "R = {:.0f} P = {:.0f} Y = {:.0f}".format(degrees(rcam[0]), degrees(rcam[1]), degrees(rcam[2]))

                # Draw rotation and translation values on the image
                cv2.putText(image, disp_text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                cv2.putText(image, "Camera position [cm]:", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                cv2.putText(image, disp_text2, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                cv2.putText(image, "Camera rotation [deg]:", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                cv2.putText(image, disp_text3, (5, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

            # Resize final image
            if prev_res is not None and prev_res[0] > 0 and prev_res[1] > 0:
                image = cv2.resize(image, prev_res)

    else:
        if show_values:
            # Text to display
            display_text1 = " - No marker detected - "
            display_text2 = "X = ? Y = ? Z = ?"
            display_text3 = "R = ? P = ? Y = ?"

            # Draw rotation and translation values on the image
            cv2.putText(image, display_text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(image, "Camera position [cm]:", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(image, display_text2, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(image, "Camera rotation [deg]:", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(image, display_text3, (5, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Resize final image
        if prev_res is not None and prev_res[0] > 0 and prev_res[1] > 0:
            image = cv2.resize(image, prev_res)

    if disp:
        # Display the final image
        cv2.imshow("Estimation result", image)
        cv2.waitKey(0)

    return image, rvec, tvec


def estimate_camera_pose_on_video(source: Union[str, int], marker_len: Union[int, float], cam_mtx: np.ndarray, dist_coefs: np.ndarray, dict_name: Optional[str] = None, show_values: bool = True, show_id: bool = False, show_axis: bool = True, src_res: Optional[Tuple[int, int]] = None, prev_res: Optional[Tuple[int, int]] = None) -> None:
    """Estimates the camera position using single ArUco marker on the video.

    Args:
        source (str or int): Path to video file or device index. Device id '0' is for built-in camera.
        marker_len (int or float): Side length of the marker in millimeter.
        cam_mtx (np.ndarray): Camera matrix.
        dist_coefs (np.ndarray): Distortion coefficients.
        dict_name (str, optional): Type of ArUco marker. Passing it may speed up the program.
        show_values (bool, optional): When it is True, the translation [cm] and rotation [deg] values are displayed.
        show_id (bool, optional): When it is True, marker's ids are displayed.
        show_axis (bool, optional): When it is True, marker's axis are displayed.
        src_res (Tuple[int, int], optional): Resolution of the captured video.
        prev_res (Tuple[int, int], optional): Resolution of the displayed video.

    Raises:
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    vc = initialize_videocapture(source, src_res)

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Loop until there are no frames left
    while rval:
        # Get the frame with markers data on it
        frame, *_ = estimate_camera_pose_on_image(frame, marker_len, cam_mtx, dist_coefs, dict_name, False, show_values, show_id, show_axis, True, prev_res)

        # Update the output image and get new frame
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        # Exit if ESC key button or X window button pressed
        key = cv2.waitKey(20)
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


def marker_and_camera_pose_on_video(source: Union[str, int], marker_len: Union[int, float], cam_mtx: np.ndarray, dist_coefs: np.ndarray, dict_name: Optional[str] = None, src_res: Optional[Tuple[int, int]] = None) -> None:
    """Debug function.

        Markers must be of the same size and type in order to assess their position correctly.

        Args:
            source (str or int): Path to video file or device index. Device id '0' is for built-in camera.
            marker_len (int or float): Side length of the marker in meters.
            cam_mtx (np.ndarray): Camera matrix.
            dist_coefs (np.ndarray): Distortion coefficients.
            dict_name (str, optional): Type of ArUco marker. Passing it may speed up the program.
            src_res (Tuple[int, int], optional): Resolution of the captured video.

        Raises:
            SystemError: If program is unable to open video source

        """
    cv2.namedWindow("Preview")
    vc = initialize_videocapture(source, src_res)

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    import time
    prev_frame_time = 0
    new_frame_time = 0
    while rval:
        # Detect aruco markers
        corners_list, *_ = detect_on_image(image=frame, dict_name=dict_name, disp=False)

        # Loop over every detected marker's corners
        for corners in corners_list:
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_len, cam_mtx, dist_coefs)
            rvec = rvec[0][0]
            tvec = tvec[0][0]
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, [corners])
            # Draw axis of the marker
            cv2.aruco.drawAxis(frame, cam_mtx, dist_coefs, rvec, tvec, marker_len/2)

            # Obtain the rotation matrix tag->camera
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            rotation_matrix_t = rotation_matrix.T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix_t)
            rot_cam = [roll, pitch, yaw]
            display_text2 = f"X = {int(tvec[0])}   Y = {int(tvec[1])}   Z = {int(tvec[2])}"
            display_text3 = f"R = {int(degrees(rot_cam[0]))}   P = {int(degrees(rot_cam[1]))}   Y = {int(degrees(rot_cam[2]))}"

            cv2.putText(frame, "Marker position:", (5, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text2, (5, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, "Marker rotation:", (5, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text3, (5, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

            pos_cam = -rotation_matrix_t * np.matrix(tvec).T
            roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix_t)
            # Get position and attitude of rhe camera respt to the marker
            display_text2 = f"X = {int(pos_cam[0])}   Y = {int(pos_cam[1])}   Z = {int(pos_cam[2])}"
            display_text3 = f"R = {int(degrees(roll))}   P = {int(degrees(pitch))}   Y = {int(degrees(yaw))}"

            cv2.putText(frame, "Camera position:", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text2, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, "Camera rotation:", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
            cv2.putText(frame, display_text3, (5, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = f"FPS: {int(fps)}"
        cv2.putText(frame, fps, (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)-140), 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 50, 255), 3, cv2.LINE_AA)

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
    # These are examples of how to use some functions. Uncomment and complete the data to run the program #

    # --- Estimate ArUco marker position and orientation on the video source --- #
    param_path = "calib_params.yml"  # Path and name of the .yml file with calibration parameters
    src = 0  # Number of camera feed
    res = (1280, 720)  # Resolution of the camera for which the calibration was performed
    mark_len = 20  # Physical side length of a single marker in mm
    dict = "DICT_4X4_50"  # Dictionary name of Aruco marker from constants.py or "None"
    mtx, dist = load_coefficients(param_path)  # Load calibration parameters
    estimate_markers_pose_on_video(source=src, marker_len=mark_len, cam_mtx=mtx, dist_coefs=dist, dict_name=dict, src_res=res)

    pass
