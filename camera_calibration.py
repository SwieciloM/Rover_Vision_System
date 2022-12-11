#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pathlib
import time

from constants import ARUCO_DICT, RESOLUTION
from typing import Tuple, List, Union, Optional
from marker_detection import detect_on_image, draw_markers_on_image


def calibrate_chessboard(path: str, board_size: Tuple[int, int], square_len: Union[int, float], image_format: str = 'jpg'):
    """Calibrate a camera using chessboard images."""
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Inner size (height, width)
    isize = (board_size[0] - 1, board_size[1] - 1)

    # Object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,6,0)
    objp = np.zeros((isize[0]*isize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:isize[1], 0:isize[0]].T.reshape(-1, 2)
    objp = objp * square_len  # Meter is a better metric

    # Arrays to store object points and image points from all the images.
    objp_list = []  # 3d point in real world space
    imgp_list = []  # 2d points in image plane

    images = pathlib.Path(path).glob(f'*.{image_format}')
    # Iterate through all images
    for fname in images:
        # Reading image and conversion to grayscale
        image = cv2.imread(str(fname))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, chess_corners = cv2.findChessboardCorners(gray, (isize[1], isize[0]), None)

        # If found, collect object points and image points (after refining them)
        if ret:
            objp_list.append(objp)
            subpix_corners = cv2.cornerSubPix(gray, chess_corners, (11, 11), (-1, -1), criteria)
            imgp_list.append(subpix_corners)

    if len(objp_list):
        # Calibrate camera
        results = cv2.calibrateCameraExtended(objp_list, imgp_list, gray.shape[::-1], None, None)
        ret, mtx, dist, rvecs, tvecs, _, _, error = results

        # Return camera matrix, distortion coefficients, rotation/translation vectors and reprojection errors
        return [ret, mtx, dist, rvecs, tvecs, error]
    else:
        raise RuntimeError("No valid images to perform calibration")


def calibrate_aruco(path: str, dict_name: str, board_size: Tuple[int, int], marker_len: Union[int, float], marker_separation: Union[int, float], image_format: str = 'jpg'):
    """Apply camera calibration using aruco.The dimensions are in mm."""
    # Verify that the supplied dict exist and is supported by OpenCV
    if ARUCO_DICT.get(dict_name, None) is None:
        raise ValueError("No such dict_name as '{}'".format(dict_name))

    # Attempt to detect the markers for the given dict
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    aruco_params = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.GridBoard_create(board_size[0], board_size[1], marker_len, marker_separation, aruco_dict)

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
    results = cv2.aruco.calibrateCameraArucoExtended(corners_list, id_list, counter, board, np.shape(image)[0:2], None, None)
    ret, mtx, dist, rvecs, tvecs, _, _, error = results

    # Return camera matrix, distortion coefficients, rotation/translation vectors and reprojection errors
    return [ret, mtx, dist, rvecs, tvecs, error]


def calibrate_charuco(path: str, dict_name: str, board_size: Tuple[int, int], marker_len: Union[int, float], square_len: Union[int, float], image_format: str = 'jpg'):
    """Apply camera calibration using aruco. The dimensions are in mm."""
    # Verify that the supplied dict exist and is supported by OpenCV
    if ARUCO_DICT.get(dict_name, None) is None:
        raise ValueError("No such dict_name as '{}'".format(dict_name))

    # Attempt to detect the markers for the given dict
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    arucoParams = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.CharucoBoard_create(board_size[0], board_size[1], square_len, marker_len, aruco_dict)

    corners_list, id_list = [], []
    image = np.zeros((1, 1))
    inner_corners_num = (board_size[0]-1)*(board_size[1]-1)
    img_dir = pathlib.Path(path)
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        print(f'Analizing {img}')
        image = cv2.imread(str(img))
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)

        resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=image, board=board)
        # If a Charuco board was found, let's collect image/corner points
        # Check the number of chessboard inner corners
        if resp == inner_corners_num:
            # Add these corners and ids to our calibration arrays
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)
            print('Image data extracted successfully.')
        else:
            print('No data extracted from the image.')

    # Actual calibration
    print("\nCalibration process started...")
    results = cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=corners_list, charucoIds=id_list, board=board, imageSize=np.shape(image)[0:2], cameraMatrix=None, distCoeffs=None)
    ret, mtx, dist, rvecs, tvecs, _, _, error = results
    print("\nCalibration completed.\n")
    # Return camera matrix, distortion coefficients, rotation/translation vectors and reprojection errors
    return [ret, mtx, dist, rvecs, tvecs, error]


def check_supported_resolutions(source: int = 0) -> None:
    """Checks which common resolutions are supported by the camera.

    Args:
        source (int): Device index. If 0, primary camera (webcam) will be used.

    Raises:
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    # Validate source argument
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        raise TypeError("Source parameter does not accept {}".format(type(source)))

    if cap is None:
        raise SystemError("Unable to open video source: {}".format(source))

    sup_res = []

    print("Searching for resolutions supported by the camera... This process can take several minutes.")
    # Looping through every resolution and adding supported ones to the list
    for width, height in RESOLUTION:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        width_set = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_set = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width == width_set and height == height_set:
            sup_res.append(f"{str(width)}x{str(height)}")

    # Printing the results
    num_sup_res = len(sup_res)
    if num_sup_res == 0:
        print("No supported resolution found.")
    elif num_sup_res == 1:
        print(f"1 supported resolution found:")
    else:
        print(f"{num_sup_res} supported resolutions found:")
    for res in sup_res:
        print(f"   {res}")

    cap.release()


def save_coefficients(mtx: np.ndarray, dist: np.ndarray, path: str) -> None:
    """Save the camera matrix and the distortion coefficients to given path/file.

    Args:
        mtx (array-like): Camera matrix.
        dist (array-like): Distortion coefficients.
        path (str): Path to file where data will be saved (.yml is best to use).

    """
    # Opening the file storage to write
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

    # Writing data
    cv_file.write('K', mtx)
    cv_file.write('D', dist)

    cv_file.release()


def load_coefficients(path: str) -> List[np.ndarray]:
    """Loads camera matrix and distortion coefficients.

    Args:
        path (str): Path to .yml file.

    """
    # Opening the file storage to read
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # Specifying the type to retrieve matrix instead of FileNode object
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def get_aruco_calimgs(board_size: Tuple[float, float], dict_name: Optional[str] = None, source: Union[str, int] = 0, resolution: Optional[Tuple[int, int]] = None, n_max: int = 20, path: str = 'images\calibration_images', image_format: str = 'jpg', min_time_inter: float = 0.5) -> None:
    """Creates and saves calibration images containing ArUco board.

    Args:
        board_size (Tuple[float]): Number of rows and columns in the currently used board.
        dict_name (str, optional): Indicates the type of ArUco markers that are placed on board.
        source (str or int, optional): Path to video file or device index. If 0, primary camera (webcam) will be used.
        resolution (Tuple[int, int], optional): Resolution of the captured video.
        n_max (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', 'png' etc.
        min_time_inter (float, optional): Time in seconds determining minimal interval between two following images.

    Raises:
        ValueError: If given dict_name or path is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    n_aruco = board_size[0]*board_size[1]

    # Formatting the file extension
    image_format = image_format.strip().strip(".").lower()

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

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    print("Images acquisition started.")
    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_max:
        # Find aruco corners
        aruco_corners, ids, _ = detect_on_image(frame, dict_name=dict_name, disp=False)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if len(aruco_corners) == n_aruco and time_inter > min_time_inter:
            if not cv2.imwrite("{}\\aruco_calib_{}.{}".format(path, n + 1, image_format), frame):
                raise ValueError("Image couldn't be saved! Check the path!")
            frame = draw_markers_on_image(frame, aruco_corners, ids)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_max}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    print("End of capture. Obtained {} out of {} images.".format(n, n_max))

    vc.release()
    cv2.destroyAllWindows()


def get_chess_calimgs(board_size: Tuple[float, float], source: Union[str, int] = 0, resolution: Optional[Tuple[int, int]] = None, n_max: int = 20, path: str = 'images\calibration_images', image_format: str = 'jpg', min_time_inter: float = 0.5) -> None:
    """Creates and saves calibration images containing chessboard.

    Args:
        board_size (Tuple[float]): Number of rows and columns in the currently used board.
        source (str or int, optional): Path to video file or device index. If 0, primary camera (webcam) will be used.
        resolution (Tuple[int, int], optional): Resolution of the captured video.
        n_max (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', '.png' etc.
        min_time_inter (float, optional): Time in seconds determining minimal interval between two following images.

    Raises:
        ValueError: If given path is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    inner_size = (board_size[0]-1, board_size[1]-1)

    # Formatting the file extension
    image_format = image_format.strip().strip(".").lower()

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

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    print("Images acquisition started.")
    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_max:
        # Find chessboard and aruco corners
        ret, inner_corners = cv2.findChessboardCorners(frame, inner_size, None)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if ret and time_inter > min_time_inter:
            if not cv2.imwrite("{}\\chess_calib_{}.{}".format(path, n + 1, image_format), frame):
                raise ValueError("Image couldn't be saved! Check the path!")
            frame = cv2.drawChessboardCorners(frame, inner_size, inner_corners, ret)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_max}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    print("End of capture. Obtained {} out of {} images.".format(n, n_max))

    vc.release()
    cv2.destroyAllWindows()


def get_charuco_calimgs(board_size: Tuple[float, float], dict_name: Optional[str] = None, source: Union[str, int] = 0, resolution: Optional[Tuple[int, int]] = None, n_max: int = 20, path: str = 'images\calibration_images', image_format: str = 'jpg', min_time_inter: float = 0.5) -> None:
    """Creates and saves calibration images containing ChArUco board.

    Args:
        board_size (Tuple[float, float]): Number of rows and columns in the currently used board.
        dict_name (str, optional): Indicates the type of ArUco markers that are placed on board.
        source (str or int, optional): Path to video file or device index. If 0, primary camera (webcam) will be used.
        resolution (Tuple[int, int], optional): Resolution of the captured video.
        n_max (int, optional): Maximum number of images to be captured.
        path (str, optional): Path to destination where images will be saved.
        image_format (int, optional): Format of images like 'jpg', '.png' etc.
        min_time_inter (float, optional): Time in seconds determining minimal interval between two following images.

    Raises:
        ValueError: If given dict_name or path is not valid
        TypeError: If given source argument has wrong type
        SystemError: If program is unable to open video source

    """
    cv2.namedWindow("Preview")
    n_aruco = (board_size[0]*board_size[1])//2
    inner_size = (board_size[0]-1, board_size[1]-1)

    # Formatting the file extension
    image_format = image_format.strip().strip(".").lower()

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

    # Counter for number of saved images and start time of capturing
    n = 0
    last_img_time = time.time()

    print("Images acquisition started.")
    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_max:
        # Find chessboard and aruco corners
        ret, inner_corners = cv2.findChessboardCorners(frame, inner_size, None)
        aruco_corners, ids, _ = detect_on_image(frame, dict_name=dict_name, disp=False)

        # Time difference between current and last frame
        time_inter = time.time() - last_img_time

        # If pattern found: save frame, change time of last record, increment saved pic counter
        if ret and len(aruco_corners) == n_aruco and time_inter > min_time_inter:
            if not cv2.imwrite("{}\\charuco_calib_{}.{}".format(path, n + 1, image_format), frame):
                raise ValueError("Image couldn't be saved! Check the path!")
            frame = cv2.drawChessboardCorners(frame, inner_size, inner_corners, ret)
            frame = draw_markers_on_image(frame, aruco_corners, ids)
            last_img_time = time.time()
            n += 1

        # Display actual number of saved images
        display_text = f"Obtained patterns: {n}/{n_max}"
        cv2.putText(frame, display_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 209, 117), 2)

        # Update the output image
        cv2.imshow("Preview", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(10)
        # Exit if ESC key button or X window button pressed
        if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    print("End of capture. Obtained {} out of {} images.".format(n, n_max))

    vc.release()
    cv2.destroyAllWindows()


def calculate_reprojected_error(path: str, board_size: Tuple[int, int], square_len: Union[int, float], image_format: str = 'jpg'):
    """Calibrate a camera using chessboard images."""
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Inner size (height, width)
    isize = (board_size[0] - 1, board_size[1] - 1)

    # Object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,6,0)
    objp = np.zeros((isize[0]*isize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:isize[1], 0:isize[0]].T.reshape(-1, 2)
    objp = objp * square_len  # Meter is a better metric

    # Arrays to store object points and image points from all the images.
    objp_list = []  # 3d point in real world space
    imgp_list = []  # 2d points in image plane

    images = pathlib.Path(path).glob(f'*.{image_format}')
    # Iterate through all images
    for fname in images:
        # Reading image and conversion to grayscale
        image = cv2.imread(str(fname))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, chess_corners = cv2.findChessboardCorners(gray, (isize[1], isize[0]), None)

        # If found, collect object points and image points (after refining them)
        if ret:
            objp_list.append(objp)
            subpix_corners = cv2.cornerSubPix(gray, chess_corners, (11, 11), (-1, -1), criteria)
            imgp_list.append(subpix_corners)

    if len(objp_list):
        # Calibrate camera
        results = cv2.calibrateCameraExtended(objp_list, imgp_list, gray.shape[::-1], None, None)
        ret, mtx, dist, rvecs, tvecs, _, _, error = results

    mean_error = 0
    for i in range(len(objp_list)):
        imgpoints2, _ = cv2.projectPoints(objp_list[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgp_list[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    return mean_error/len(objp_list)


if __name__ == '__main__':
    # Load coefficients
    # mtx, dist = load_coefficients('calibration_chess.yml')
    # original = cv2.imread('images\\calibration_images\\calib50.jpg')
    # dst = cv2.undistort(original, mtx, dist, None, mtx)
    # cv2.imwrite('images\\calibration_images\\undist_chess.png', dst)

    #get_charuco_calimgs(board_size=(5, 5), dict_name="DICT_4X4_50", resolution=(1280, 720), n_max=50, path="C:/Users/micha/Pulpit/test", min_time_inter=0.5)
    # ret, mtx, dist, rvecs, tvecs, error = calibrate_charuco("images/calibration_images/charuco", "DICT_5X5_50", (5, 7), 23, 30, "png")
    # print(f"ret:\n{ret}\n")
    # print(f"Camera matrix:\n{mtx}\n")
    # print(f"Distortion coefficients:\n{dist}\n")
    # print(f"Rotation vectors:\n{rvecs}\n")
    # print(f"Translation vectors:\n{tvecs}\n")
    # print(f"Average re-projection error:\n{sum(error)/len(error)}\n")
    # Save coefficients into a file
    # save_coefficients(mtx, dist, "calib_charuco_realsense_1280x720.yml")

    #get_aruco_calimgs(board_size=(5, 7), dict_name="DICT_6X6_1000", resolution=(1280, 720), n_max=50, path="images/calibration_images/aruco", min_time_inter=0.5)
    # ret, mtx, dist, rvecs, tvecs, error = calibrate_aruco("images/calibration_images/aruco", "DICT_6X6_50", (7, 5), 26, 3, "png")
    # print(f"ret:\n{ret}\n")
    # print(f"Camera matrix:\n{mtx}\n")
    # print(f"Distortion coefficients:\n{dist}\n")
    # print(f"Rotation vectors:\n{rvecs}\n")
    # print(f"Translation vectors:\n{tvecs}\n")
    # print(f"Average re-projection error:\n{sum(error)/len(error)}\n")
    # Save coefficients into a file
    # save_coefficients(mtx, dist, "calib_aruco_realsense_1280x720.yml")

    #get_aruco_calimgs(board_size=(7, 5), dict_name="DICT_6X6_50", source=2, resolution=(1280, 720), n_max=100, path="images/calibration_images/aruco", image_format="png", min_time_inter=1)
    # ret, mtx, dist, rvecs, tvecs, error = calibrate_chessboard("images/calibration_images/chess/TestUltimate", (6, 9), 30, "png")
    # print(f"ret:\n{ret}\n")
    # print(f"Camera matrix:\n{mtx}\n")
    # print(f"Distortion coefficients:\n{dist}\n")
    # # print(f"Rotation vectors:\n{rvecs}\n")
    # # print(f"Translation vectors:\n{tvecs}\n")
    # print(f"Average re-projection error:\n{sum(error)/len(error)}\n")
    # # Save coefficients into a file
    # save_coefficients(mtx, dist, "test4.yml")

    # # Load coefficients
    # base = cv2.imread('images/calibration_images/chess/chess_calib_1.png')
    # cv2.imshow("Oryginał", base)
    # for file in ["calib_charuco_realsense_1280x720.yml", "calib_aruco_realsense_1280x720.yml", "calib_chess_realsense_1280x720.yml"]:
    #     mtx, dist = load_coefficients(file)
    #     undst = cv2.undistort(base, mtx, dist, None, mtx)
    #     cv2.imshow(file, undst)
    #     #cv2.imwrite('images\\calibration_images\\undist_chess.png', dst)

    # Load coefficients
    base = cv2.imread('images/calibration_images/chess/chess_calib_1.png')
    cv2.imshow("Oryginał", base)
    for file in ["calib_chess_realsense_1280x720.yml", "test4.yml", "test3.yml"]:
        mtx, dist = load_coefficients(file)
        undst = cv2.undistort(base, mtx, dist, None, mtx)
        cv2.imshow(file, undst)
        #cv2.imwrite('images\\calibration_images\\undist_chess.png', dst)

    cv2.waitKey(0)

# TODO: Print logi sygnalizujące obecny stan wykonywania kalibracji
# TODO: Stworzyć test sprawdzający jakość kalibracji
# TODO: Uporządkować kod i dopisać komentarze/docstringi
# TODO: Zrobić zabezpieczenia przed pustymi folderami, i zdjęciami które nie nadają się do kalibracji
# TODO: Zrobić system sprawdzający jakość kalibracji i usuwające zjęcia negatywnie wpływające na nią
# TODO: Naprawić funkcję kalibrującą 'calibrate_aruco()'
# TODO: Dodoać argparsera z możliwością wyboru danej funkcji

# TODO: Poprawienie sposobu zapisu wszystkich funkcji tak aby działały również na innych systemach

# TODO: Optymalizacja funkcji get_charuco_calimgs
#       Podczas detekcji tablicy na obrazie w wysokiej rozdzielczości zbyt wiele zasobów jest zużywanych na wykrycie
#       samej szachownicy przez co wartość FPS znacząco maleje. Potencjalne  rozwiązanie to downsizing przetważanego
#       obrazu, wykonanie na nim detekcji a następnie przeskalowanie do normalnego rozmiaru. Źródło:
#       https://stackoverflow.com/questions/15018620/findchessboardcorners-cannot-detect-chessboard-on-very-large-images-by-long-foca/15074774#15074774
