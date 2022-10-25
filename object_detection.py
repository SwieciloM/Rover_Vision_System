#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

cv2.namedWindow("Preview")

# Check data type of source argument
vc = cv2.VideoCapture(0)

# Try to get the first frame
if (vc is not None) and vc.isOpened():
    rval, frame = vc.read()
else:
    raise SystemError("Unab")

while rval:
    # Update the output image
    cv2.imshow("Preview", frame)
    rval, frame = vc.read()

    key = cv2.waitKey(20)
    # Exit if ESC key button or X window button pressed
    if key == 27 or cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
        break