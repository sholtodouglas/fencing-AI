
## Cuts the videos into a set of short clips where each actual hit happens. These clips are used by the data_labeller to 
## label the clips where the referee had to distinguish whos priority.
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

# Prelim info, FOTR light box is frame[329:334, 380:500]
# therefore FOTL box is frame[329:334, 140:260]
# FOTL OFF-TARGET frame[337:348, 234:250]
# FOTR OFF-TARGET frame[337:348, 390:406]
#      scoreleft = frame[310:325, 265:285]
#      scoreRight = frame[310:325, 355:375]
    
FFMPEG_BIN = "ffmpeg"

## Load a pretrained logisitc classifier which distinguishes between numbers 0-15.
import cPickle
with open('logistic_classifier_0-15.pkl', 'rb') as fid:
    
    model = cPickle.load(fid)



fps = str(13)
jump_length = 260 # this is how long our 'recording time' will be, where we don't check for lights, actual recording time, its so long because we want to skip people testing their blades after hits
## is jump length - hide length = 'clip length'
hide_length = 200 # where we're not actually interested in keeping the frames, but don't want them to be seen by 'not in record mode'
video_number = 0

videos_to_cut = 0
for i in os.listdir(os.getcwd() + "/precut"):
    if i.endswith(".mp4"):
        videos_to_cut = videos_to_cut + 1


print "Cutting", videos_to_cut, "videos"

already_processed = 400
for vid in os.listdir(os.getcwd() + "/precut"):
    if vid.endswith(".mp4") and int(vid.replace(".mp4","")) >= already_processed :
        print "Video:", vid
        clips_recorded = 0
        recording_mode = False
    
        cap = cv2.VideoCapture("precut/" + str(vid))
        
        cap_end_point = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print "Length of Vid:", cap_end_point, int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        cap.release()
        cap_end_point = cap_end_point - jump_length # ensures videos don't overrun
        print "Beginning to cut..."
        
        position = 2000

        while position < cap_end_point:
            cap = cv2.VideoCapture("precut/" + str(vid))

            print position, "big while loop", int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            
            if position == cap_end_point:
                print "should not be here"
                break
            

            
            if recording_mode == True:
                print "Recording Mode On"
                output_file = 'videos/' + str(vid).replace(".mp4","") +"-" + str(clips_recorded) + '.mp4'
                
           
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

                frames_till_video_end = jump_length
                
                proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

            
            if cap.isOpened():
                
                cap.set(1,position)
                cap.set(cv2.cv.CV_CAP_PROP_FPS, 10000)

                while cap.isOpened():
                    ret,frame = cap.read()
                    position = position + 1

                    

                    if recording_mode == False:
                        
                        if position % 100 == 0:
                            print position
                            print cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                        # this check is here because the vid should be prevented from 
                        # starting a clip less than frames_till_vid_end away from 
                        # the end
                        if position == (cap_end_point):
                            break
                        elif cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >= cap_end_point:
                            print "break"
                            position = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                            break
                        try:
                            if (np.sum(abs(frame[337:348, 234:250].astype(int)-white_box.astype(int))) <= 7000) or (np.sum(abs(frame[337:348, 390:406].astype(int)-white_box.astype(int))) <= 7000) or (np.sum(abs(frame[330:334, 380:500].astype(int)-green_box.astype(int))) <= 40000) or (np.sum(abs(frame[330:334, 140:260].astype(int)-red_box.astype(int))) <= 40000):
                                
                                left_score = model.predict(frame[309:325, 265:285].reshape(1,-1))
                                right_score = model.predict(frame[309:325, 355:375].reshape(1,-1))
                                print left_score, right_score
                        
                                if (left_score == 15) or (right_score == 15):
                                    print "dont record this hit"
                                    position = position + 25
                                    break
                                elif (left_score == 0) and (right_score == 0):
                                    print "dont record this hit"
                                    position = position + 25
                                    break
                                else:
                                    print "recording hit"
                        
                                    ## jump back 50 frames to the action of the hit
                                    position = position - 50
                                    
                                    print "Light seen, position-", position
                                    recording_mode = True
                                    break
                        except:
                            break
                    
                    
                    if recording_mode == True:
                        if frames_till_video_end >= hide_length:
                            if position % 2 == 0:

                                proc.stdin.write(frame.tostring())
                        
                        frames_till_video_end = frames_till_video_end - 1
                        if frames_till_video_end == 0:
                            print "finished clip"
                            recording_mode = False
                            proc.stdin.close()
                            proc.stderr.close()
                            print(clips_recorded)
                            clips_recorded = clips_recorded+1
                            break   
            else:
                print("Failed to open video")

            cap.release()
            video_number = video_number + 1





