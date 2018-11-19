# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 01:43:29 2018

@author: Abdullah
"""

import cv2

import os

path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDs\AliNeg'
dir="HogDs/Ali"
for x in range(38):
    
        file = dir + "/" + "alithr ("+str(x) + ").png"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        #pic=cv2.imread(dir + "/" + "L1 ("+str(x)+").png")
        image=cv2.flip( image, 1 )
    
        cv2.imwrite(os.path.join(path , "aliright"+str(x)+'.png'),image )