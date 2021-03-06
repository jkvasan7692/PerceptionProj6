import os,sys
import numpy as np
import math
from sklearn import svm
import random as rng
import pickle

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv

#%%
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


def read_training_images():
    print_debug("Reading Train Images")
    labels_list = [45,21,38,35,17,1,14]
    training_directory = './Training/'
    training_data = []
    for dirname,dirnames,filenames in os.walk(training_directory):
        for dir_name in dirnames:
            label_dir_str = dir_name[-5:]
            #if int(label_dir_str) in labels_list:
            for filename in os.listdir(training_directory+dir_name):
                if ".ppm" in filename:
                    image_path = training_directory+dir_name+'/'+filename
                    image = cv.imread(image_path,0)
                    label_image_tup = (image,int(label_dir_str))
                    training_data.append(label_image_tup)
    return training_data


def read_test_images():
    print_debug("Reading Test Image")
    labels_list = [45,21,38,35,17,1,14]
    training_directory = './Testing/'
    test_data = []
    for dirname,dirnames,filenames in os.walk(training_directory):
        for dir_name in dirnames:
            label_dir_str = dir_name[-5:]
            #if int(label_dir_str) in labels_list:
            for filename in os.listdir(training_directory+dir_name):
                if ".ppm" in filename:
                    image_path = training_directory+dir_name+'/'+filename
                    image = cv.imread(image_path,0)
                    label_image_tup = (image,int(label_dir_str))
                    test_data.append(label_image_tup)
    return test_data


def get_hog_features_labels(training_data):
    print_debug("Getting Hog Features")
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
        #print_debug(hog_features.shape)
        hog_features_list.append(hog_features)
        labels_list.append(label)
    hog_array = np.array(hog_features_list)
    return hog_array,labels_list


def get_hog_features(images_list):
    print_debug("Getting Hog Features")
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
    hog_features_list = []
    for image in images_list:
        resized_image = cv.resize(image,(64,64))
        hog_features = hog.compute(resized_image)
        #print_debug(hog_features.shape)
        hog_features_list.append(hog_features)
    hog_array = np.array(hog_features_list)
    return hog_array


def train_svm(hog_features,labels):
    label_array = np.array(labels)
    hog_features = hog_features.reshape((hog_features.shape[0],hog_features.shape[1]))
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(hog_features,label_array)
    print_debug("Training SVM")
 
    return clf

def test_svm(svm,hog_features,labels,test_images):
    label_array = np.array(labels)
    hog_features = hog_features.reshape((hog_features.shape[0],hog_features.shape[1]))
    decision=svm.predict(hog_features)
    classifications = np.array_equal(label_array,decision)
    accuracy=0
    for i in range(0,decision.shape[0]):
        if decision[i]==labels[i]:
            accuracy=accuracy+1
    print_debug(float(accuracy)/decision.shape[0])

def find_sign_board(svm,hog_features_array):
    print_debug("Hog Features Array: ", hog_features_array)
    hog_features_array = hog_features_array.reshape((hog_features_array.shape[0],hog_features_array.shape[1]))
    decision = svm.predict(hog_features_array)
    labels_list = [45,21,38,35,17,1,14]
    mask = np.zeros((decision.shape[0]))
    for i in range(0,decision.shape[0]):
        if decision[i] in labels_list:
            mask[i]= 1
    return mask


# for file_name in files_list:
#     if ".jpg" in file_name:
#         image = cv.imread(images_dir+file_name)
#         #final = np.hstack((image, white_balance_loops(image)))
#         thresh_image = hsv_thresh(image)
#         show(image,"image",0)
def hsv_thresh(image):
    hsv_image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    sat_channel = hsv_image[:,:,1]
    ret,thresholded_image = cv.threshold(sat_channel,110,255,cv.THRESH_BINARY)
    show(thresholded_image,"thresh",10)

    return thresholded_image
    
def findListOfImages(inpImg , contours, hierarchy):
    boundedRectImages = list() 
    filteredRect = list()       
        
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        if boundRect[i][2]*boundRect[i][3] >1000 and  boundRect[i][2]*boundRect[i][3] < 10000:
            xStart = boundRect[i][0]
            yStart = boundRect[i][1]
            xEnd = boundRect[i][0]+boundRect[i][2]
            yEnd = boundRect[i][1]+boundRect[i][3]
            
            roi = inpImg[yStart:yEnd , xStart:xEnd]
            
            filteredRect.append(boundRect[i])
#            show(roi , "roi", 0)
            boundedRectImages.append(roi)
            
#            cv.rectangle(inpImg, (int(boundRect[i][0]), int(boundRect[i][1])),(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (0,255,0), 2)
            
    return boundedRectImages, filteredRect,  inpImg
    
def drawBoundingBoxForSign(img , boundRectList, mask):

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    
    for i in range(len(boundRectList)):
        if(mask[i] == 1):
            boundRect = boundRectList[i]
            cv.rectangle(img, (int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,0), 2)

#%%
files_list = os.listdir(images_dir)
files_list.sort()
#train_data = read_training_images()
#train_hog_features,labels = get_hog_features_labels(train_data)
#svm_trained = train_svm(train_hog_features,labels)

#%%
#pkl_file = "svm_params.pkl"
#s = pickle.dumps(svm_trained)
##print_debug("Dumps:", s)
#pickle.dump(s , pkl_file)
#
#k = pickle.load(pkl_file)
##print_debug("Loaded: ", k)
#trainedParams = pickle.loads(k)
#print_debug("Trained_Params", trainedParams)

with open('svm_params.pkl' , 'rb') as file:
    svm_trained = pickle.load(file)
    
#test_data = read_test_images()
#test_hog_features,test_labels = get_hog_features_labels(test_data)
#test_svm(svm_trained,test_hog_features,test_labels,test_data)
    
#%%
for file_name in files_list:
     if ".jpg" in file_name:
         image = cv.imread(images_dir+file_name)
        
#         detector_params = cv.SimpleBlobDetector_Params()
#         detector = cv.SimpleBlobDetector(detector_params)
        
#         detector_params.filterByArea = True
#         detector_params.minArea = 100
         thresh_image = hsv_thresh(image)
        
      
         contours, hierarchy = cv.findContours(thresh_image , cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
        
         boundedRoi, filteredBoundRect, imgWithContours = findListOfImages(image , contours , hierarchy)
         print_debug("Filtered Bounded Rect:", filteredBoundRect)
         mask = np.ones(len(boundedRoi))
         
         if(len(boundedRoi) > 0):
        
 ##%% Call to the HOG and SVM function
             hog_array = get_hog_features(boundedRoi)
    #%%
             mask = find_sign_board(svm_trained , hog_array)
             
         
             
         cv.waitKey()
         print("SVM_mask: ", mask)
        
 #%% Draw the mask filtered contours
         drawBoundingBoxForSign(image , filteredBoundRect, mask)

    

         show(image,"image",10)


