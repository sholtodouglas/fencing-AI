## Looks at each clip, if there were two lights, look at the next clip to see what changes there were in the scoreboard.
## Depending on the changes in the scoreboard, label the clip according. If there weren't two lights, disregard the clip 
## as it doesn't provide as any more information. 

import cv2
import tensorflow as tf
import numpy as np
import argparse
import time
import cv
import subprocess as sp
import os
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
green_box = cv2.imread("greenbox.png")
red_box = cv2.imread("redbox.png")
white_box = cv2.imread("whitebox.png")


import cPickle
with open('logistic_classifier_0-15.pkl', 'rb') as fid:
    
    model = cPickle.load(fid)

################################################################################################

def check_lights(frame):
    # returns a string, either On-On, On-Off, Off-On, Off-Off, On-No, No-On, Off-No, No-Off
    
    # red is on the left, green on the right
    leftOff = False
    leftOn = False
    rightOff = False
    rightOn = False
    string = ""
     #check for left on target light
    if (np.sum(abs(frame[330:334, 140:260].astype(int)-red_box.astype(int))) <= 40000):
        string = string + "On"
    #check for left off target light
    elif (np.sum(abs(frame[337:348, 234:250].astype(int)-white_box.astype(int))) <= 7000):
        string = string + "Off"
    else:
        string = string + "No"
        
    #check for right off target light
    string = string + "-"
    if (np.sum(abs(frame[330:334, 380:500].astype(int)-green_box.astype(int))) <= 40000):
    
       string = string + "On"
    #ccheck for right on target light
    elif (np.sum(abs(frame[337:348, 390:406].astype(int)-white_box.astype(int))) <= 7000):
        string = string + "Off"
    else:
        string = string + "No"
        
    return string
 ################################################################################################

def check_score(frame):
     left = model.predict(frame[309:325, 265:285].reshape(1,-1)) 
     right = model.predict(frame[309:325, 355:375].reshape(1,-1))
     return left, right
################################################################################################

def caption(hit_type,left,right,update_left,update_right):
    caption = "None"
    if hit_type == "On-On":
        if update_left-left == 1 and update_right-right == 0:
            caption = "L"
        if update_left-left == 0 and update_right-right == 1:
            caption = "R"
        if update_left-left == 0 and update_right-right == 0:
            caption = "T"
    if hit_type == "On-Off":
        if update_left-left == 1 and update_right-right == 0:
            caption = "L"
        if update_left-left == 0 and update_right-right == 0:
            caption = "R"
    if hit_type == "Off-On":
        if update_left-left == 0 and update_right-right == 1:
            caption = "R"
        if update_left-left == 0 and update_right-right == 0:
            caption = "L"
            
    return caption

################################################################################################

for i in os.listdir(os.getcwd() + "/videos"):
    if i.endswith(".mp4"): 
        #print i.split("-")
        match_number = int(i.split("-")[0])
        hit_number = int(i.split("-")[1].replace(".mp4",""))
        print "Match-Hit",match_number, hit_number
        cap = cv2.VideoCapture("videos/" + i)
        cap_end_point = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.set(1,cap_end_point-1)  
        ret,frame = cap.read()
        hit_type = check_lights(frame)
        left,right = check_score(frame)
        print hit_type
        print left,right
        
        cap.release()

        if hit_type == "On-On" or hit_type == "On-Off" or hit_type == "Off-On":
        #### now open the following hit
            next_hit = "videos/" + str(match_number) + "-" + str(hit_number+1) + ".mp4"
            if os.path.isfile(next_hit) == True:
                # open the next hit
                cap = cv2.VideoCapture(next_hit)
                cap.set(1,0)
                ret,frame = cap.read()
                update_left,update_right = check_score(frame)
                cap.release()
                priority = caption(hit_type,left,right,update_left,update_right)
                print update_left, update_right
                
                print priority
                if priority != 'None':
                    os.rename("videos/"+ i, "training_quarantine/"+priority+i)
                print " "

        continue
    else:
        continue


# And we're done, data collected!