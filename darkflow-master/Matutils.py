# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 01:30:52 2018

@author: Abdullah
"""

import os
import cv2
import numpy as np

# Return all the image filenames in a given dir
def getImageFileList(path):
    #TODO: probably an os function that returns full path+filename
    fullPathList = []
    filelist = os.listdir(path)
    for f in filelist:
        if(f.upper()).endswith('png'):
            fullPathList.append(path + f)
    return fullPathList

# Load all images and return list
def getImagesFromPath(imageFiles, width, height, maxfiles=0):
    imgList = []
    for i in imageFiles:
        if(maxfiles!=0 and len(imgList) >= maxfiles):
            break;
        img = cv2.imread(i)
        img = cv2.resize(img, (width, height))
        img = np.float32(img)
        imgList.append(img)
    return np.asarray(imgList)

# Load random list of images and return list
def getRandomImages(imageFiles, count, width, height):
    import random
    randomFiles = random.sample(imageFiles, count) 
    imgList = []
    for f in randomFiles:
        img = cv2.IMREAD_GRAYSCALE(f)#, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height))
        img = np.float32(img)
        imgList.append(img)
    return np.asarray(imgList)

#set current process to run at low priority
def setProcessLowPriority():
    import win32api,win32process,win32con
    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS) 

def getImageFileCountFromPath(path):
    counter=0
    filelist = os.listdir(path)
    for f in filelist:
        if(f.upper()).endswith('png'):
            counter=counter+1
    return counter