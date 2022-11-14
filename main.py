#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from constants import ARUCO_DICT
from pose_estimation import is_rotation_matrix, rotation_matrix_to_euler_angles
from camera_calibration import load_coefficients
import cProfile, pstats, io
from math import degrees, sqrt, atan2
from typing import Optional, Union, Tuple
from marker_detection import detect_on_image

def profile(fnc):

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def print_hi(name):
    path = r"C:\Users\micha\Pulpit\Test_aruco\WIN_20221111_19_35_26_Pro.jpg"
    img = cv2.imread(path)
    mtx, dist = load_coefficients('calibration_chess_1280x720.yml')


    @profile
    def estimate_camera_pose_on_image(image: np.ndarray, marker_len: Union[int, float], cam_mtx: np.ndarray,
                                      dist_coefs: np.ndarray, dict_name: Optional[str] = None, disp: bool = False,
                                      show_values: bool = True, show_id: bool = False, show_axis: bool = True,
                                      ret_final: bool = False, prev_res: Optional[Tuple[int, int]] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
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
                    cv2.aruco.drawAxis(image, cam_mtx, dist_coefs, rvec, tvec, marker_len / 2)

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
                    disp_text2 = "X = {} Y = {} Z = {}".format(int(pcam[0] / 10), int(pcam[1] / 10), int(pcam[2] / 10))
                    disp_text3 = "R = {:.0f} P = {:.0f} Y = {:.0f}".format(degrees(rcam[0]), degrees(rcam[1]),
                                                                           degrees(rcam[2]))

                    # Draw rotation and translation values on the image
                    cv2.putText(image, disp_text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                    cv2.putText(image, "Camera position [cm]:", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117),
                                2)
                    cv2.putText(image, disp_text2, (5, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)
                    cv2.putText(image, "Camera rotation [deg]:", (5, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117),
                                2)
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

    estimate_camera_pose_on_image(img, 105, mtx, dist, "DICT_5X5_100")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
