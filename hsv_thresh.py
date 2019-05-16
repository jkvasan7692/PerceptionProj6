# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:31:19 2019

@author: Gautam Balachandran
"""

import os,sys
import numpy as np
import math
from sklearn import svm

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

def read_training_images():
    labels_list = [45,21,38,35,17,1,14]
    training_directory = '../Training/'
    training_data = []
    for dirname,dirnames,filenames in os.walk(training_directory):
        for dir_name in dirnames:
            label_dir_str = dir_name[-5:]
            if int(label_dir_str) in labels_list:
                for filename in os.listdir(training_directory+dir_name):
                    if ".ppm" in filename:
                        image_path = training_directory+dir_name+'/'+filename
                        image = cv.imread(image_path,0)
                        label_image_tup = (image,int(label_dir_str))
                        training_data.append(label_image_tup)
    return training_data


def get_hog_features_labels(training_data):
    win_size=(64,64)
    block_size= (32,32)
    cell_size= (8,8)
    block_stride = (8,8)
    nbins = 18
    hog = cv.HOGDescriptor(_winSize=win_size,
                            _blockSize=block_size,
                            _blockStride=block_stride,
                            _cellSize=cell_size,
                            _nbins=nbins,
                            _signedGradient=True)
    labels_list = []
    hog_features_list = []
    for data in training_data:
        image = data[0]
        label = data[1]
        resized_image = cv.resize(image,(64,64))
        hog_features = hog.compute(resized_image)
        print_debug(hog_features.shape)
        hog_features_list.append(hog_features)
        labels_list.append(label)
    hog_array = np.array(hog_features_list)
    return hog_array,labels_list


def train_svm(hog_features,labels):
    label_array = np.array(labels)
    hog_features = hog_features.reshape((hog_features.shape[0],hog_features.shape[1]))
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(hog_features,label_array)
    print_debug(clf)







files_list = os.listdir(images_dir)
files_list.sort()
train_data = read_training_images()
hog_features,labels = get_hog_features_labels(train_data)
svm_trained = train_svm(hog_features,labels)

# for file_name in files_list:
#     if ".jpg" in file_name:
#         image = cv.imread(images_dir+file_name)
#         #final = np.hstack((image, white_balance_loops(image)))
#         thresh_image = hsv_thresh(image)
#         show(image,"image",0)


