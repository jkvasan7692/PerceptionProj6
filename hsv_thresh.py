# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:31:19 2019

@author: Gautam Balachandran
"""

import os,sys
import numpy as np
import math

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv


images_dir = "../input/"
testing_dir = "../Testing/"
training_dir = "../Training/"
DEBUG = True

def print_debug(*objects):
    if DEBUG:
        print(*objects)

def show(image,window_name,wait):
    print_debug('Displaying '+str(window_name))
    cv.imshow(window_name, image)
    cv.waitKey(wait)
    #cv.destroyAllWindows()

def white_balance_loops(img):
    result = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    for x in range(result.shape[0]):
        for y in range(result.shape[1]):
            l, a, b = result[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
            result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
    return result


def hsv_thresh(image):
	hsv_image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
	sat_channel = hsv_image[:,:,1]
	ret,thresholded_image = cv.threshold(sat_channel,120,255,cv.THRESH_BINARY)
	show(thresholded_image,"thresh",0)



files_list = os.listdir(images_dir)
files_list.sort()
for file_name in files_list:
    if ".jpg" in file_name:
        image = cv.imread(images_dir+file_name)
        #final = np.hstack((image, white_balance_loops(image)))
        thresh_image = hsv_thresh(image)

        show(image,"image",0)


