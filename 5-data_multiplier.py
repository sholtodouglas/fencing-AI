## Performs data-augmentation by flipping all clips horizontally. 

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



FFMPEG_BIN = "ffmpeg"
import subprocess as sp
import os
fps = str(13)


######################### Here we flip every video horizontally, and double our training set! ########################
for i in os.listdir(os.getcwd() + "/training_quarantine"):
    if i.endswith(".mp4"):
    	cap = cv2.VideoCapture("training_quarantine/" + str(i))
        if i[0] == 'L':
            i = 'R'+ i.lstrip('L')
        elif i[0] == 'R':
            i = 'L'+ i.lstrip('R')

        output_file = 'more_training_data/' + str(i).replace('.mp4', '-flipped') + '.mp4' 
            # clips_recorded = clips_recorded+1
        cap.set(cv2.cv.CV_CAP_PROP_FPS, 10000)
        command = [FFMPEG_BIN,
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '640*360',
        '-pix_fmt', 'bgr24',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        output_file ]

            # this is how long our video will be.
        
            
        proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        print("more diagnostics")
            
        counter = 0
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            counter = counter + 1
            if ret==True:

#flip the frame horizontally
                frame = cv2.flip(frame,1)
                proc.stdin.write(frame.tostring())
        

    
            else:
                break
                
                
        proc.stdin.close()
        print("stderr")
        proc.stderr.close()
        print("successful")

        # Release everything if job is finished
        cap.release()

############################################################################################################################
        
