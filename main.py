#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
from constants import ARUCO_DICT


def print_hi(name):
    dict_name = "DICT_7X7_1000"

    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_name])
    print(aruco_dict.bytesList)
    print("--------------------------")
    print(len(aruco_dict.bytesList))

    def none_or_str(value):
        if value == 'None':
            return None
        return value



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
