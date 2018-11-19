# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 17:44:55 2018

@author: Abdullah
"""
import cv2
import ssl
import numpy as np
from skimage import data, exposure
from skimage.feature import hog
from skimage import data, exposure
from skimage.feature import hog
from darkflow.net.build import TFNet
import time
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import urllib








def roidetect(w,x,y,z):
       
       if w>100:
           w=w-100
       else:    
       
           if w>50:
               w=w-50
           else:    
               if w>25:
                   w=w-25
               else:    
                   if w>15:
                       w=w-15
                   else:    
                       if w>10:
                           w=w-10    
                   
                   
       
       if 1280-x>100:
           x=x+100
       else:    
        
           if 1280-x>50:
               x=x+50
           else:
               if 1280-x>25:
                   x=x+25
       if y>100:
           y=y-100
       else:   
            if y>50:
               y=y-50
            else:    
              if y<25:
                  y=y-25
              else:    
                  if y<15:
                      y=y-15
                  else:    
                      y=y
       if 720-z>100:
           z=z+100
       else:            
           if 720-z>50:
               z=z+50
           else:
               if 720-z>25:
                   z=z+25
               else:
                   if 720-z>15:
                       z=z+15
                   else:
                       if 720-z>10:
                           z=z+10
                       else:
                           if 720-z>10:
                               z=z+10
                           
                   
                   
           
       roi=w,x,y,z
       return roi   









model_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\bacha\model.h5'
model_weights_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\bacha\weights.h5'

label="no"
#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDs\Today'
   
        
def predict(file):
      fixed_size = tuple((150, 150)) 
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
      sc=model.predict_proba(x)
      sc = np.max(model.predict(x))
      return answer,sc    
    
    
def kam(signal):
        personcheck=0
        count=0
        options = {
        'model': 'cfg/tiny-yolo.cfg',
        'load': 'bin/tiny-yolo.weights',
        'threshold': 0.2,
        'gpu': 1.0
        }
        MHI_DURATION = 20
        DEFAULT_THRESHOLD = 32
        timestamp = 0
        ch=0
        fc=1
        count=0
        personcheck=0
        sz=1280
        szr=720
        tfnet = TFNet(options)
        nf=150
        f=2
        url="https://192.168.100.4:8080"
        #capture = cv2.VideoCapture("Dataset/ToDay/vid ("+str(f)+").mp4")
        #capture = cv2.VideoCapture("http://192.168.100.4:4747/video")
        capture = cv2.VideoCapture("http://192.168.100.2:4747/video")


        #capture = cv2.VideoCapture("Dataset/Testing.mp4")
        iskasignal=signal
        #if iskasignal==1:
        #capture = cv2.VideoCapture("Dataset/vid (1).mp4")
            #capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        taw=10
        while True:
            stime = time.time()
            #ssl._create_default_https_context = ssl._create_unverified_context
#            imgResp=urllib.request.urlopen(url)
 #           imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
  #          image=cv2.imdecode(imgNp,-1)
            ret, image = capture.read()
            
            #image=img.copy()
            #image=cv2.flip( image, 1 )
        
            
            
            if ret:
                
                cv2.imshow('original', image)
               
                rows,cols = image.shape[:2]
        
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
                            
                if personcheck==1: 
                        bach= image.copy()
                        
                        
                        
                        #cv2.imshow('found', nm)  
                        sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                        
                        cv=roidetect(tl[1],br[1],tl[0],br[0])
                        
                        frame=image[cv[0]:cv[1],cv[2]:cv[3]].copy()
                        
                        h, w = frame.shape[:2]
                        if count==0:
                            ret, image = capture.read()
                            #image=cv2.imdecode(imgNp,-1)
                            sarakam=(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                            prev_frame = frame.copy()
                            frame=image[cv[0]:cv[1],cv[2]:cv[3]].copy()
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
                   
                        sarakam[cv[0]:cv[1],cv[2]:cv[3]]=mh.copy()
                        count =10
             #********************************Start****HOG****************               
                        
                        fd, hog_image = hog(mh, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True,transform_sqrt=False, feature_vector=True)
            
                       
                        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                        hogch=hog_image_rescaled*255
                        hog_img_uint8 = hogch.astype(np.uint8)
                  
                        
                        
                        
                        cv2.imwrite(os.path.join(path , 'me.png'),hog_img_uint8 )
                        hg=cv2.imread(str(path)+"\me.png")
                        #hg=cv2.cvtColor(np.array(hg), cv2.COLOR_BGR2RGB)
                        res,score = predict(hg)
                        if res == 1:
                            lab="throw"
                            nm = cv2.rectangle(bach, (cv[3],cv[1]), (cv[2],cv[0]), (0,0,255), 5)
                        elif res == 0:
                            lab="notthrow"
                            nm = cv2.rectangle(bach, (cv[3],cv[1]), (cv[2],cv[0]), (34,139,34), 5)
                     
            
                
                        print(lab)
                        
                       
                        cv2.putText(nm, lab, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                        print(score*100)
                        cv2.imshow('found', nm) 
                        cv2.imshow("HOG",hog_image_rescaled)
                            
                       
                        prev_frame = frame.copy()  
                        
                        if timestamp%50==0:
                            personcheck=0
                            count=0
#                            cv2.destroyWindow("found") 
                            cv2.destroyWindow("HOG")
                            #nm = cv2.rectangle(0, 0, 0, 2, 5)
                            lab=""
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            ch=1
                            
                           
                            break
                #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                if ch==1:                
                                
                            
                                           
        
                    return lab,mh,hog_image_rescaled 
                    capture.release()
                    
                    
                    break
   
    
res,lab,h=kam(1)