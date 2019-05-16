import os,sys
import numpy as np
import math
import random as rng

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv


images_dir = "./input/"
testing_dir = "./Testing/"
training_dir = "./Training/"
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
    ret,thresholded_image = cv.threshold(sat_channel,110,255,cv.THRESH_BINARY)
    show(thresholded_image,"thresh",10)

    return thresholded_image
    
def findListOfImages(inpImg , contours, hierarchy):
    boundedRectImages = list()        
        
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        if boundRect[i][2]*boundRect[i][3] >3000 and  boundRect[i][2]*boundRect[i][3] < 10000:
            xStart = boundRect[i][0]
            yStart = boundRect[i][1]
            xEnd = boundRect[i][0]+boundRect[i][2]
            yEnd = boundRect[i][1]+boundRect[i][3]
            
            roi = inpImg[yStart:yEnd , xStart:xEnd]
#            show(roi , "roi", 0)
            boundedRectImages.append(roi)
            
#            cv.rectangle(inpImg, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
            
    return boundedRectImages, inpImg
    
def drawBoundingBoxForSign(img , contours, mask):

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    
    for i, c in enumerate(contours):
        if(mask[i] == 1):
            
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            cv.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)

files_list = os.listdir(images_dir)
files_list.sort()
for file_name in files_list:
    if ".jpg" in file_name:
        image = cv.imread(images_dir+file_name)
        
        detector_params = cv.SimpleBlobDetector_Params()
        detector = cv.SimpleBlobDetector(detector_params)
        
        detector_params.filterByArea = True
        detector_params.minArea = 100
       thresh_image = hsv_thresh(image)
        
      
        contours, hierarchy = cv.findContours(thresh_image , cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
        
        boundedRoi, imgWithContours = findListOfImages(image , contours , hierarchy)
        
        mask = np.ones(len(boundedRoi))
        
#%% Call to the HOG and SVM function
        
#%% Draw the mask filtered contours
#        drawBoundingBoxForSign(image , contours, mask)

    

        show(imageKeyp,"image",10)


