# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:16:37 2018

@author: Abdullah
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 02:10:15 2018

@author: Abdullah
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os
from matplotlib import pyplot as plt

from skimage import data, exposure




from skimage.feature import hog




options = {
    'model': 'cfg/tiny-yolo.cfg',
    'load': 'bin/tiny-yolo.weights',
    'threshold': 0.2,
    'gpu': 1.0
}

tfnet = TFNet(options)
nf=560
f=3
capture = cv2.VideoCapture("Dataset/vid ("+str(f)+").mp4")


capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
MHI_DURATION = 20
DEFAULT_THRESHOLD = 32
timestamp = 0
ch=0
fc=1
count=0
personcheck=0
sz=1280
szr=720
def roidetect(w,x,y,z):
       
       if w>100:
           w=w-100
       else:    
       
           if w>50:
               w=w-50
           else:    
               if w<25:
                   w=w-25
       
       if sz-x>100:
           x=x+100
       else:    
        
           if sz-x>50:
               x=x+50
           else:
               if sz-x>25:
                   x=x+25
       if y>100:
           y=y-100
       else:   
            if y>50:
               y=y-50
            else:    
              if y<25:
                  y=y-25 
       if szr-z>100:
           z=z+100
       else:            
           if szr-z>50:
               z=z+50
           else:
               if szr-z>25:
                   z=z+25       
           
       roi=w,x,y,z
       return roi   




taw=10
while True:
    stime = time.time()
    ret, image = capture.read()
    #image=cv2.flip( image, 1 )
#    num_rows, num_cols = image.shape[:2]
#    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), -90, 1)
#    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    
    if ret==False:
        break
    
    if ret:
        szr,sz=image.shape[:2]
         #cv2.imshow('original', image)
        path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDS'
        if personcheck==0:
            results = tfnet.return_predict(image)
            phlibar=0
            
            for  result in results:
                    label = result['label']
                    
                    if label =="person" :
                        confidence = result['confidence']
                        if phlibar==0:
                            pc=result['confidence']
                        if confidence>=pc:
                            phlibar=1
                            tl = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'], result['bottomright']['y'])
                            
                            confidence = result['confidence']
                            pc=confidence
                            personcheck=1
                            #break
                    else:
                        break
                    
            
            # text = '{}: {:.0f}%'.format(label, confidence * 100)
            
            # image=pic.copy()
            
            
        if personcheck==1:
                bach= image.copy()
                cv2.imshow('original', image)
                nm = cv2.rectangle(bach, tl, br, 2, 5)
                
                cv2.imshow('found', nm)  
                sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                personcheck=1
                cv=roidetect(tl[1],br[1],tl[0],br[0])
                frame=image[cv[0]:cv[1],cv[2]:cv[3]].copy()
                
                
                
                # fog=frame
                #if timestamp==0:
                
               
                
       #check point dehan se         
            
                # ret, image = capture.read()
                #sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
#                fc+=1
#                if fc%20==0:
#                    personcheck=0
            
                
             #   frame=image[cv[0]:cv[1],cv[2]:cv[3]].copy()
                h, w = frame.shape[:2]
                if count==0:
                    ret, image = capture.read()
                    sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                    prev_frame = frame.copy()
                    frame=image[cv[0]:cv[1],cv[2]:cv[3]].copy()
                    motion_history = np.zeros((h, w), np.float32)
                    h,w=image.shape[:2]
                    sarakam = np.zeros((h, w), np.float32)
                    
                    
                    # Gradient X y wala kam
                    
                    count=9
                    
                    
                    
                frame_diff = (cv2.absdiff(frame, prev_frame)).copy()
                gray_diff = (cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)).copy()
                ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
                timestamp += 1    
      # update motion history
                cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)       
      # normalize motion history
                mh = (np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)).copy()
                # r,c=mh.shape[:2]
                
                            
               # gx=filters.sobel_h(im)
                #gy=filters.sobel_v(im)
                
                gx = cv2.Sobel(mh, cv2.CV_32F, 1, 0, ksize=1)
                gy = cv2.Sobel(mh, cv2.CV_32F, 0, 1, ksize=1)
                
                sarakam[cv[0]:cv[1],cv[2]:cv[3]]=mh.copy()
                count =10
 #********************************Start****HOG****************               
                
                fd, hog_image = hog(mh, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True,transform_sqrt=False, feature_vector=True)

                #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
                
#                ax1.axis('off')
#                ax1.imshow(image, cmap=plt.cm.gray)
#                ax1.set_title('Input image')
                
                # Rescale histogram for better display
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                hogch=hog_image_rescaled*255
                hog_img_uint8 = hogch.astype(np.uint8)
                #hog_img_inv = cv2.bitwise_not(hog_image)
                
#                ax2.axis('off')
#                ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#                ax2.set_title('Histogram of Oriented Gradients')
#                plt.show()
                               
#********************************End****HOG****************
#           
                
                mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                cv2.imshow("MHI", mh)
                
                cv2.imshow("HOG",hog_image_rescaled)
                
                options = {
                'model': 'cfg/throw.cfg',
                'load': 'ckpt/throw-125',
                'threshold': 0.2,
                'gpu': 1.0
                }
                backtorgb = cv2.cvtColor(hog_image_rescaled,cv2.COLOR_GRAY2RGB)
                results = tfnet.return_predict(backtorgb)
                result=results(0)
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                            
                confidence = result['confidence']
                bach= backtorgb.copy()
                cv2.imshow('original', backtorgb)
                nm = cv2.rectangle(bach, tl, br, 2, 5)
                
                cv2.imshow('found', nm) 
                
                
                
                
#                cv2.imshow("DX",gx)
#                cv2.imshow("Dy",gy)
#                cv2.imshow("mag",mag)
#                cv2.imshow("ang",angle)
#                cv2.imshow("Sara",sarakam)
               
                prev_frame = frame.copy()  
                if timestamp>=nf:
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    ch=1
                    
                    
                    break
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if ch==1:                
            break            
                    
                                   

          
capture.release()
cv2.destroyAllWindows()           