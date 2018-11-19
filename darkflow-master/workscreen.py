# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'working_screen.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

# import Opencv module
import cv2
import os
import numpy as np
from keras.preprocessing.image import   img_to_array
from keras.models import  load_model
import time
#from matplotlib import pyplot as plt
import cv2

from darkflow.net.build import TFNet



path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDs\Today'

from skimage import  exposure
from skimage.feature import hog
import winsound
#from darkflow.net.build import TFNet
#import numpy as np
#import time
#import os

#import urllib

#from skimage import data

start = time.time()
lab="notthrow"
#url='http://192.168.1.11:8080//shot.jpg'

#Define Path
model_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\modelRGB.h5'
model_weights_path = r'C:\Users\Abdullah\Downloads\FYP K Khate\Using Keras\models\weightsRGB.h5'

label="no"
#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
path = r'C:\Users\Abdullah\Downloads\FYP K Khate\darkflow-master\darkflow-master\HogDs\Today'

img_width, img_height = 150, 150
fixed_size = tuple((150, 150))
MHI_DURATION = 20
DEFAULT_THRESHOLD = 32
timestamp = 0
ch=0
fc=1
count=0
personcheck=0
sz=1280
szr=720
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000

#Prediction Function
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
  sc=model.predict_proba(x)
  sc = np.max(model.predict(x))
  return answer,sc












options = {
    'model': 'cfg/tiny-yolo.cfg',
    'load': 'bin/tiny-yolo.weights',
    'threshold': 0.2,
    'gpu': 1.0
}

tfnet = TFNet(options)
nf=150
f=2

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
               if w>25:
                   w=w-25
               else:    
                   if w>15:
                       w=w-15
                   else:    
                       if w>10:
                           w=w-10    
                   
                   
       
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
              else:    
                  if y<15:
                      y=y-15
                  else:    
                      y=y
       if szr-z>100:
           z=z+100
       else:            
           if szr-z>50:
               z=z+50
           else:
               if szr-z>25:
                   z=z+25
               else:
                   if szr-z>15:
                       z=z+15
                   else:
                       if szr-z>10:
                           z=z+10
                       else:
                           if szr-z>10:
                               z=z+10
                           
                   
                   
           
       roi=w,x,y,z
       return roi   



ret=True
taw=10





















class Ui_work(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 30, 640, 480))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(680, 30, 271, 481))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(830, 530, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.controlTimer)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 530, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.timer = QTimer()
        #set timer timeout callback function
        #self.timer.timeout.connect(self.viewCam)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Main"))
        self.label_2.setText(_translate("MainWindow", "HOG"))
        self.pushButton.setText(_translate("MainWindow", "Start"))
        self.pushButton_2.setText(_translate("MainWindow", "Back"))
        
        
    def viewCam(self):
        # read image in BGR format
        ret, image = self.cap.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #wind2=hog
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.label.setPixmap(QPixmap.fromImage(qImg))
        #self.label_2.setPixmap(Qpixmap.fromImage())

    # start/stop timer
    def controlTimer(self):
        timestamp = 0
        ch=0
        
        count=0
        personcheck=0
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture("Dataset/testvid.mp4")
            
            while True:
                stime = time.time()
            #    imgResp=urllib.request.urlopen(url)
            #    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            #    image=cv2.imdecode(imgNp,-1)
                ret, image = self.cap.read()
                throwcount=0
                lab="notthrow"
                #cv2.imshow('original', image)
                #image=img.copy()
                #image=cv2.flip( image, 1 )
            
            #    if ret==False:
            #        break
                
                if ret:
                   
                    rows,cols = image.shape[:2]
            
                    if  personcheck==0:
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
                                ret, image =self.cap.read()
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
                            if timestamp%20==0:
                                fd, hog_image = hog(mh, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualise=True,transform_sqrt=False, feature_vector=True)
                
                           
                                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                                hogch=hog_image_rescaled*255
                                hog_img_uint8 = hogch.astype(np.uint8)
                          
                                
                                
                                
                                cv2.imwrite(os.path.join(path , 'me.png'),hog_img_uint8 )
                                hg=cv2.imread(str(path)+"\me.png")
                                res,score = predict(hg)
                                if res == 1:
                                    lab="throw"
                                    nm = cv2.rectangle(bach, (cv[3],cv[1]), (cv[2],cv[0]), (0,0,255), 5)
                                    cv2.putText(nm, lab, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                                    #cv2.imshow('ThrowDetected', nm)
                                    winsound.Beep(frequency, duration)
                                    throwcount+=1
                                elif res == 0:
                                    lab="notthrow"
                                    nm = cv2.rectangle(bach, (cv[3],cv[1]), (cv[2],cv[0]), (34,139,34), 5)
                                    cv2.putText(nm, lab, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                                    
                                    #cv2.imshow('found', nm)
                                        
                                
                            
                                
                                
                                
                                cv2.destroyWindow("HOG")
                                #nm = cv2.rectangle(0, 0, 0, 2, 5)
                                lab="notthrow"
                                height, width, channel = hg.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(hg, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                self.label_2.setPixmap(QPixmap.fromImage(qImg))
                                
                            else:
                                  nm = cv2.rectangle(bach, (cv[3],cv[1]), (cv[2],cv[0]), (34,139,34), 5)
                                  cv2.putText(nm, lab, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                                  #cv2.imshow('found', nm)
                            if timestamp%50==0:
                                personcheck=0
                                count=0
                                #cv2.destroyWindow("found")
                 #********************************Start****HOG****************               
                            
                           
                            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            #wind2=hog
                            # get image infos
                            
                            
                            
                            nm=cv2.cvtColor(nm, cv2.COLOR_BGR2RGB)
                            height, width, channel = nm.shape
                            step = channel * width
                            # create QImage from image
                            qImg = QImage(nm.data, width, height, step, QImage.Format_RGB888)
                            # show image in img_label
                            self.label.setPixmap(QPixmap.fromImage(qImg))
                            #cv2.imshow('found', nm) 
                            #cv2.imshow("HOG",hog_image_rescaled)
                                
                           
                            prev_frame = frame.copy()  
                            if timestamp%50==0:
                                personcheck=0
                                count=0
                                
                                #nm = cv2.rectangle(0, 0, 0, 2, 5)
                                lab=""
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                ch=1
                                
                                
                                break
                    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                    if ch==1:                
                        break            
            
            
            
            
            
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.pushButton.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.pushButton.setText("Start")  
    #def capture(self):
        
        #set control_bt callback clicked  function
        
                
            
            
            


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_work()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

