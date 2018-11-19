# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 03:57:25 2018

@author: Abdullah
"""
#from MuhafizzSupporting import kam 
import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import  QApplication, QMainWindow,QWidget,QLabel
from PyQt5.uic import loadUi
from PyQt5 import QtGui,QtCore
import cv2


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 640
        self.height = 480
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        #create a label
        label = QLabel(self)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                         QtGui.QImage.Format_RGB888)
        convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
        pixmap = QPixmap(convertToQtFormat)
        resizeImage = pixmap.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
        QApplication.processEvents()
        label.setPixmap(resizeImage)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())