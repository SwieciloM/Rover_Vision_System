#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pathlib
import time

from marker_detection import detect_on_image


def calibrate_chessboard(dir_path, image_format, square_size, width, height):
    """Calibrate a camera using chessboard images."""
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height]

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    images = pathlib.Path(dir_path).glob(f'*.{image_format}')
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

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def calibrate_aruco(dirpath, image_format, marker_length, marker_separation):
    """Apply camera calibration using aruco.The dimensions are in cm."""
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
    arucoParams = cv2.aruco.DetectorParameters_create()
    board = cv2.aruco.GridBoard_create(5, 7, marker_length, marker_separation, aruco_dict)

    counter, corners_list, id_list = [], [], []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*.{image_format}'):
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img_gray,
            aruco_dict,
            parameters=arucoParams
        )

        if first == 0:
            corners_list = corners
            id_list = ids
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        first = first + 1
        counter.append(len(ids))

    counter = np.array(counter)
    # Actual calibration
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_list,
        id_list,
        counter,
        board,
        img_gray.shape,
        None,
        None
    )
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


def get_charuco_calimgs(board_size, source=0, n_img=20, dest_path='images\calibration_images', image_format='jpg', min_time_inter=0.5):
    cv2.namedWindow("Preview")

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

    n_aruco = (board_size[0]*board_size[1])//2
    inner_size = (board_size[0]-1, board_size[1]-1)
    n = 0
    last_img_time = time.time()
    # Loop until there are no frames left or the required number of images has been reached
    while rval and n < n_img:
        # Find the chess board corners
        ret, inner_corners = cv2.findChessboardCorners(frame, inner_size, None)
        aruco_corners, _, _ = detect_on_image(frame, dict_name='DICT_4X4_50', disp=False)

        time_inter = time.time() - last_img_time

        # If found, add object points, image points (after refining them)
        if ret and len(aruco_corners) == n_aruco and time_inter > min_time_inter:
            # Save frame
            cv2.imwrite("{}\\calib{}.{}".format(dest_path, n+1, image_format), frame)

            # Draw the corners
            frame = cv2.drawChessboardCorners(frame, inner_size, inner_corners, ret)

            n += 1

            last_img_time = time.time()

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

    get_charuco_calimgs(board_size=(5, 5), n_img=50, min_time_inter=0.25)

    #image = cv2.imread('images/calibration_images/cali.jpg')
    ret, mtx, dist, rvecs, tvecs = calibrate_charuco("images\\calibration_images", '.jpg', 23, 30)
    print(f"ret:\n {ret}")
    print(f"mtx:\n {mtx}")
    print(f"dist:\n {dist}")
    print(f"rvecs:\n {rvecs}")
    print(f"tvecs:\n {tvecs}")

    # Save coefficients into a file
    save_coefficients(mtx, dist, "calibration_charuco3.yml")

    # Load coefficients
    mtx, dist = load_coefficients('calibration_charuco3.yml')
    original = cv2.imread('images\\calibration_images\\calib50.jpg')
    dst = cv2.undistort(original, mtx, dist, None, mtx)
    cv2.imwrite('images\\calibration_images\\undist_charuco.png', dst)

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
