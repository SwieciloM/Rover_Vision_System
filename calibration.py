#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pathlib
import time

from constants import ARUCO_DICT
from typing import Tuple, Union
from marker_detection import detect_on_image, draw_markers_on_image


def calibrate_chessboard(path, board_size: Tuple[int, int], square_size: Union[int, float], image_format: str = 'jpg'):
    """Calibrate a camera using chessboard images."""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    height = board_size[0]-1
    width = board_size[1]-1

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = pathlib.Path(path).glob(f'*.{image_format}')
    # Iterate through all images
    for fname in images:
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    print(np.shape(gray)[0:2])
    print(gray.shape[::-1])
    print(np.shape(objpoints))
    print(np.shape(imgpoints))
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def calibrate_aruco(path: str, dict_name, board_size: Tuple[int, int], marker_length: Union[int, float], marker_separation: Union[int, float], image_format: str = 'jpg'):
    """Apply camera calibration using aruco.The dimensions are in mm."""
    # Verify that the supplied dict exist and is supported by OpenCV
    if ARUCO_DICT.get(dict_name, None) is None:
        raise ValueError("No such dict_name as '{}'".format(dict_name))

    # Attempt to detect the markers for the given dict
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    aruco_params = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.GridBoard_create(board_size[0], board_size[1], marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    image = np.zeros((1, 1))
    img_dir = pathlib.Path(path)
    first = True
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*.{image_format}'): # Tu trzeba dać sprawdzanie czy ta lokalizacja nie będzie pusta
        image = cv2.imread(str(img))
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

        if first:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))

        counter.append(len(corners))

    counter = np.array(counter)
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(corners_list, id_list, counter, board, np.shape(image)[0:2], None, None)
    return [ret, mtx, dist, rvecs, tvecs]


def calibrate_charuco(dirpath, image_format, marker_length, square_length):
    """Apply camera calibration using aruco. The dimensions are in cm."""
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(5, 5, square_length, marker_length, aruco_dict)
    arucoParams = cv2.aruco.DetectorParameters_create()

    counter, corners_list, id_list = [], [], []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        print(f'using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_gray,
            aruco_dict,
            parameters=arucoParams
        )

        resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        if resp > 15: #20
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)

    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_list,
        charucoIds=id_list,
        board=board,
        imageSize=img_gray.shape,
        cameraMatrix=None,
        distCoeffs=None)

    # Return camera matrix, distortion coefficients, rotation and translation vectors
    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', mtx)
    cv_file.write('D', dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    """Loads camera matrix and distortion coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def get_aruco_calimgs(board_size, dict_name=None, source=0, n_img=20, path='images\calibration_images', image_format='jpg', min_time_inter=0.5) -> None:
    """Creates and saves calibration images containing ArUco board.

    Args:
        board_size (Tuple[float]): Number of rows and columns in the currently used board.
        dict_name (str, optional): Indicates the type of ArUco markers that are placed on board.
        source (str or int): Path to video file or device index. If 0, primary camera (webcam) will be used.
        n_img (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', 'png' etc.
        min_time_inter (float): Time in seconds determining minimal interval between two following images.

    Raises:
        ValueError: If given dict_name is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    n_aruco = board_size[0]*board_size[1]

    # Check data type of source argument
    if isinstance(source, int) or isinstance(source, str):
        vc = cv2.VideoCapture(source)
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_img:
        # Find aruco corners
        aruco_corners, ids, _ = detect_on_image(frame, dict_name=dict_name, disp=False)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if len(aruco_corners) == n_aruco and time_inter > min_time_inter:
            cv2.imwrite("{}\\aruco_calib_{}.{}".format(path, n + 1, image_format), frame)
            frame = draw_markers_on_image(frame, aruco_corners, ids)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_img}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


def get_chess_calimgs(board_size, source=0, n_img=20, path='images\calibration_images', image_format='jpg', min_time_inter=0.5) -> None:
    """Creates and saves calibration images containing ChArUco board.

    Args:
        board_size (Tuple[float]): Number of rows and columns in the currently used board.
        source (str or int): Path to video file or device index. If 0, primary camera (webcam) will be used.
        n_img (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', 'png' etc.
        min_time_inter (float): Time in seconds determining minimal interval between two following images.

    Raises:
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    inner_size = (board_size[0]-1, board_size[1]-1)

    # Check data type of source argument
    if isinstance(source, int) or isinstance(source, str):
        vc = cv2.VideoCapture(source)
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_img:
        # Find chessboard and aruco corners
        ret, inner_corners = cv2.findChessboardCorners(frame, inner_size, None)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if ret and time_inter > min_time_inter:
            cv2.imwrite("{}\\chess_calib_{}.{}".format(path, n + 1, image_format), frame)
            frame = cv2.drawChessboardCorners(frame, inner_size, inner_corners, ret)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_img}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


def get_charuco_calimgs(board_size, dict_name=None, source=0, n_img=20, path='images\calibration_images', image_format='jpg', min_time_inter=0.5) -> None:
    """Creates and saves calibration images containing ChArUco board.

    Args:
        board_size (Tuple[float]): Number of rows and columns in the currently used board.
        dict_name (str, optional): Indicates the type of ArUco markers that are placed on board.
        source (str or int): Path to video file or device index. If 0, primary camera (webcam) will be used.
        n_img (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', 'png' etc.
        min_time_inter (float): Time in seconds determining minimal interval between two following images.

    Raises:
        ValueError: If given dict_name is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    n_aruco = (board_size[0]*board_size[1])//2
    inner_size = (board_size[0]-1, board_size[1]-1)

    # Check data type of source argument
    if isinstance(source, int) or isinstance(source, str):
        vc = cv2.VideoCapture(source)
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    # Try to get the first frame
    if (vc is not None) and vc.isOpened():
        rval, frame = vc.read()
    else:
        raise SystemError("Unable to open video source: {}".format(source))

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_img:
        # Find chessboard and aruco corners
        ret, inner_corners = cv2.findChessboardCorners(frame, inner_size, None)
        aruco_corners, _, _ = detect_on_image(frame, dict_name=dict_name, disp=False)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if ret and len(aruco_corners) == n_aruco and time_inter > min_time_inter:
            cv2.imwrite("{}\\charuco_calib_{}.{}".format(path, n + 1, image_format), frame)
            frame = cv2.drawChessboardCorners(frame, inner_size, inner_corners, ret)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_img}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Load the input image fom disk
    # path = 'real_images//test5.jpg'
    # dict = "DICT_4X4_50"
    # image = cv2.imread(path)

    #get_aruco_calimgs(board_size=(7, 5), n_img=50, path="images\\calibration_images\\1")
    # ret, mtx, dist, rvecs, tvecs = calibrate_aruco("images\\calibration_images\\1", "DICT_6X6_50", (7, 5), 26.5, 3)
    # print(f"ret:\n {ret}")
    # print(f"mtx:\n {mtx}")
    # print(f"dist:\n {dist}")
    # print(f"rvecs:\n {rvecs}")
    # print(f"tvecs:\n {tvecs}")
    # # Save coefficients into a file
    # save_coefficients(mtx, dist, "calibration_aruco.yml")

    # get_chess_calimgs(board_size=(8, 7), n_img=50, path="images\\calibration_images\\2")
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard("images\\calibration_images\\2", (7, 8), 24)
    print(f"ret:\n {ret}")
    print(f"mtx:\n {mtx}")
    print(f"dist:\n {dist}")
    print(f"rvecs:\n {rvecs}")
    print(f"tvecs:\n {tvecs}")
    # Save coefficients into a file
    save_coefficients(mtx, dist, "calibration_chess.yml")

    get_charuco_calimgs(board_size=(5, 5), n_img=50, path="images\\calibration_images\\3")
    ret, mtx, dist, rvecs, tvecs = calibrate_charuco("images\\calibration_images\\3", '.jpg', 23, 30)
    print(f"ret:\n {ret}")
    print(f"mtx:\n {mtx}")
    print(f"dist:\n {dist}")
    print(f"rvecs:\n {rvecs}")
    print(f"tvecs:\n {tvecs}")
    # Save coefficients into a file
    save_coefficients(mtx, dist, "calibration_charuco.yml")


    # Load coefficients
    # mtx, dist = load_coefficients('calibration_chess.yml')
    # original = cv2.imread('images\\calibration_images\\calib50.jpg')
    # dst = cv2.undistort(original, mtx, dist, None, mtx)
    # cv2.imwrite('images\\calibration_images\\undist_chess.png', dst)

    # mean_error = 0
    # for i in range(len(objpoints)):
    #     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    #     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    #     mean_error += error
    # print("total error: {}".format(mean_error / len(objpoints)))

# TODO: Stworzyć test sprawdzający jakość kalibracji
# TODO: Sparametryzować i udokumentować funkcje
# TODO: Sprawdzić różnicę w obrazach przed undistortowaniem i po
# TODO: Zbadać czemu minimalna liczba wewnętrznyvh rogów w calibrate_charuco() wynosiła 20
# TODO: Naprawićć warningi o 'rererece before a assigment'
# TODO: Zrozumieć co się dzieje w funkcjch
# TODO: Uporządkować pliki
