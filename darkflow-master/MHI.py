# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:59:02 2018

@author: Abdullah
"""

import cv2
import numpy as np
import time
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import imutils

MHI_DURATION = 20
#capture=cv2.VideoCapture("http://192.168.100.4:4747/video")
#capture = cv2.VideoCapture(0)
qcapture = cv2.VideoCapture("Dataset/hog.mp4")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
taw=10
timestamp=0
count=0
DEFAULT_THRESHOLD = 32
pathformhi=""
pathfordiff=""
model_path = './models/modelRGB.h5'
model_weights_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\weightsRGB.h5'
model_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\modelRGB.h5'
#model_weights_path = './models/weightsRGB.h5'

label="no"
#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDs\Today'
fixed_size = tuple((150, 150))  
        
def predict(file):
  #x=cv2.imread(str(file))  
 # x = Image.fromarray(file)
  #x = load_img(file, target_size=(img_width,img_height))
  x = cv2.resize(np.array(file), fixed_size)
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  ans = array[0]
  #print(result)
  answer = np.argmax(ans)
 
  return answer
while True:
            stime = time.time()
            #ssl._create_default_https_context = ssl._create_unverified_context
#            imgResp=urllib.request.urlopen(url)
 #           imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
  #          image=cv2.imdecode(imgNp,-1)
            ret, image = capture.read()
           # image = imutils.rotate(image, -90)
            if ret==False:
                break
            cv2.imshow('original', image)
            #image=img.copy()
            #image=cv2.flip( image, 1 )
        
        #    if ret==False:
        #        break
        
            frame=image
            h, w = frame.shape[:2]
            if count==0:
                            ret, image = capture.read()
                            #image=cv2.imdecode(imgNp,-1)
                            sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                            prev_frame = frame.copy()
                            frame=image
                            motion_history = np.zeros((h, w), np.float32)
                            h,w=image.shape[:2]
                            sarakam = np.zeros((h, w), np.float32)
                            
                      
                            
                            count=9
            frame_diff = (cv2.absdiff(frame, prev_frame)).copy()
            gray_diff = (cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)).copy()
            ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
            timestamp += 1    
          # update motion history
            cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)       
          # normalize motion history
            mh = (np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)).copy()
            cv2.imshow("Diff",gray_diff)
            cv2.imshow("mhi",mh)
            cv2.imwrite(os.path.join(path , 'me.png'),mh)
            hg=cv2.imread(str(path)+"\me.png")
            
            prev_frame = frame.copy()
            
            result = predict(hg)
            if result == 1:
                label="throw"
            elif result == 0:
                label="notthrow"
 
    
            
            cv2.putText(frame, label, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            cv2.imshow("dndn",frame)            
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                            ch=1
                            
                           
                            break
            
                        
cv2.destroyAllWindows()                        
capture.release()