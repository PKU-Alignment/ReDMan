#!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import sys
from ezgripper_libs.ezgripper_interface import EZGripper
from PyQt4 import QtGui, QtCore

rospy.init_node('hello_ezgripper')
gripper = EZGripper('ezgripper/main')

class GripperGUI(QtGui.QMainWindow):

   def __init__(self):
      super(GripperGUI, self).__init__()
      self.initUI()

   def initUI(self):

      calibrateButton=QtGui.QPushButton("Calibrate",self)
      calibrateButton.resize(100,30)
      calibrateButton.clicked.connect(gripper.calibrate)
      #QObject.connect(calibrateButton, SIGNAL("clicked()"), gripper.calibrate)
      calibrateButton.move(50,10)
      calibrateButton.show()

      releaseButton=QtGui.QPushButton("Release",self)
      releaseButton.resize(200,200)
      releaseButton.clicked.connect(gripper.release)
      #QObject.connect(hard_closeButton, SIGNAL("clicked()"), gripper.hard_close)
      releaseButton.move(50,50)

      hard_closeButton=QtGui.QPushButton("Hard Close",self)
      hard_closeButton.resize(200,200)
      hard_closeButton.clicked.connect(self.submit_goto_hard_close) 
      hard_closeButton.move(250,50)

      openButton=QtGui.QPushButton("Open", self)
      openButton.clicked.connect(self.submit_goto_open) 
      openButton.resize(200,200)
      openButton.move(450,50)

      gotoButton=QtGui.QPushButton("0% Torque Mode", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto1) 
      gotoButton.move(50,250)

      gotoButton=QtGui.QPushButton("10%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto2) 
      gotoButton.move(150,250)

      gotoButton=QtGui.QPushButton("20%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto3) 
      gotoButton.move(250,250)

      gotoButton=QtGui.QPushButton("30%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto4) 
      gotoButton.move(350,250)

      gotoButton=QtGui.QPushButton("40%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto5) 
      gotoButton.move(450,250)

      gotoButton=QtGui.QPushButton("50%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto6) 
      gotoButton.move(550,250)

      gotoButton=QtGui.QPushButton("60%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto7) 
      gotoButton.move(150,450)

      gotoButton=QtGui.QPushButton("70%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto8) 
      gotoButton.move(250,450)

      gotoButton=QtGui.QPushButton("80%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto9) 
      gotoButton.move(350,450)

      gotoButton=QtGui.QPushButton("90%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto10) 
      gotoButton.move(450,450)

      gotoButton=QtGui.QPushButton("100%", self)
      gotoButton.resize(100,200)
      gotoButton.clicked.connect(self.submit_goto11) 
      gotoButton.move(550,450)
 
      self.statusBar()

      self.setGeometry(300, 200, 800, 850)
      self.setWindowTitle("EZGripper GUI")
      self.show()

   def submit_goto_hard_close(self):

      gripper.goto_position(0, 100)

   def submit_goto_open(self):

      gripper.goto_position(1, 100)

   def submit_goto1(self):

      gripper.goto_position(0, 10)

   def submit_goto2(self):

      gripper.goto_position(.10, 100)

   def submit_goto3(self):

      gripper.goto_position(.20, 100)

   def submit_goto4(self):

      gripper.goto_position(.30, 100)

   def submit_goto5(self):

      gripper.goto_position(.40, 100)

   def submit_goto6(self):

      gripper.goto_position(.50, 100)

   def submit_goto7(self):

      gripper.goto_position(.60, 100)

   def submit_goto8(self):

      gripper.goto_position(.70, 100)

   def submit_goto9(self):

      gripper.goto_position(.80, 100)

   def submit_goto10(self):

      gripper.goto_position(.90, 100)

   def submit_goto11(self):

      gripper.goto_position(1.0, 100)

   def submit_goto12(self):

      gripper.goto_position(.20, 100)

   def submit_goto13(self):

      gripper.goto_position(.20, 100)

   def submit_goto14(self):

      gripper.goto_position(.20, 100)

def main():

   ezgripper_app=QtGui.QApplication(sys.argv)
   ex=GripperGUI()
   sys.exit(ezgripper_app.exec_())

if __name__== '__main__':
   main()
