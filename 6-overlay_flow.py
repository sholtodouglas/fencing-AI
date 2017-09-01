# Calculate optical flow, overlay it on the clip and save it as matrices instead of the mp4 files we've been using so far.

import os
import hickle as hkl
import numpy as np
import argparse
import cv
import cv2
import matplotlib.pyplot as plt 
import subprocess as sp



def label_to_one_hot(label):
    if label == 'L':
        return (1,0,0)
    elif label == 'T':
        return (0,1,0)
    elif label == 'R':
        return (0,0,1)


def writeOpticalFlowToVideo(video_string):
    

    cap = cv2.VideoCapture(video_string)
    cap.set(cv2.cv.CV_CAP_PROP_FPS, 10000)
    cap.set(1,0)
    print cap.grab()
    
    ret, frame1 = cap.read()
    height = frame1.shape[0]
    print height
    width = frame1.shape[1]
    depth = frame1.shape[2]
    height_end = height - height/7 ## we don't really want to include the fencers names/countries
    print height_end
    height_start = 0
    
    frame1 = np.concatenate((frame1[height_start:height_end,0:width,0:depth],frame1[height_end+height/14-10:height,0:width,0:depth]), axis = 0)
    print frame1.shape
    
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    print hsv.shape, "hsv shape"
    hsv[...,1] = 255
    fps = str(13)
    FFMPEG_BIN = "ffmpeg"
    
    output_file = 'optical_flow/' + video_string.replace('final_training_clips/',"").replace('.mp4',"") +'move'+ '.mp4'
   # in the below command is set the output
   ## note the frame dimensions!
    command = [FFMPEG_BIN,
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', '640*345',
                '-pix_fmt', 'bgr24',
                '-r', fps,
                '-i', '-',
                '-an',
                '-vcodec', 'mpeg4',
                '-b:v', '5000k',
                output_file ]

    frames_till_video_end = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    cap.set(1,0)
    if frames_till_video_end == 23:
        #read a frame
        last_frame = 2
    else:
        last_frame = 1

    print frames_till_video_end
    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    cv2.imshow("frame",frame1)
    subcount = 1

    while subcount <= frames_till_video_end-last_frame:
        
        print subcount
        ret, frame2 = cap.read()
        #frame2 = frame2[int(height_start):height_end,0:width,0:depth]
        frame2 = np.concatenate((frame2[height_start:height_end,0:width,0:depth],frame2[height_end+height/14-10:height,0:width,0:depth]), axis = 0)
        #print frame2.shape
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
        # flow.shape is 240, 640, 2. (hori and verti)
        # flow is made up of floats.
        #print flow

        # take the average of the flow vectors to comepensate for camera pan.
        # this is really key. 
        flow[:,:,0] = flow[:,:,0] - np.mean(flow[:,:,0])
        flow[:,:,1] = flow[:,:,1] - np.mean(flow[:,:,1])
        print flow.shape
        
            
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        #cv2.imshow('flow',bgr)
        output = frame2
        #find the greyscale version
        gray_image = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # make a 3d grayscale version so we can overlay the color information.
        output[:,:,0] = gray_image
        output[:,:,1] = gray_image
        output[:,:,2] = gray_image
        ## set the transparency of the overlay
        alpha = 0.45
        cv2.addWeighted(bgr, alpha, output, 1 - alpha,
        0, output)
        

        #proc.stdin.write(bgr.tostring())
        proc.stdin.write(output.tostring())
        #cv2.imshow('frame2',output)
        ###cv2.imshow('frame2',bgr)
        

        output = output.reshape(-1,output.shape[0],output.shape[1],output.shape[2])
        if subcount == 1:
            to_save = output
        if subcount > 1:
            
            to_save = np.concatenate((to_save,output), axis = 0)

        subcount = subcount+1

        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',bgr)
        
        ### Here we set prvs to next so that the frame we're calculating difference from changes.
        prvs = next

    proc.stdin.close()
    proc.stderr.close() 
    cap.release()
    cv2.destroyAllWindows()
    #add an extra dimension so that they are easily added only this dimension
    #to_save = to_save.reshape(-1,output.shape[0],output.shape[1],output.shape[2],output.shape[3])
    to_save = np.expand_dims(to_save, axis=0)
    label = video_string.split('/')[1][0]
    label = np.expand_dims(label_to_one_hot(label),axis=0)

    print label
    print to_save.shape, "to save shape"
    return to_save,label



#######################################################################################################################################
data_created = 0
data_saved = 0 
data_saved_previously = []
for i in os.listdir(os.getcwd() + "/preinception_data"):  
    if i.endswith(".hkl"):
        number = i.replace(".hkl",'').split('-')[1]
        data_saved_previously.append(number)
if len(data_saved_previously) > 0:
    data_saved = int(max(data_saved_previously))
data_saved = data_saved + 1
print "Largest Number Found", data_saved
#######################################################################################################################################


for i in os.listdir(os.getcwd() + "/final_training_clips"):
    if i.endswith(".mp4"):
        print i
        output,label = writeOpticalFlowToVideo("final_training_clips/" + i)
        os.rename("final_training_clips/"+ i, "final_training_clips/already_optical_flowed/"+i)
        if data_created == 0:
            train_set = output
            train_labels = label
        else:
            train_set = np.concatenate((train_set,output), axis = 0)
            train_labels = np.concatenate((train_labels,label), axis = 0)

        data_created = data_created + 1
        if data_created % 100 == 0:

            hkl.dump(train_set, 'preinception_data/train_set-' + str(data_saved) + '.hkl', mode='w', compression='gzip', compression_opts=9)
            hkl.dump(train_labels,'final_training_data/train_labels-' + str(data_saved) + '.hkl', mode='w', compression='gzip', compression_opts=9)
            
            
            print '################### DATA SAVED', data_saved
            data_saved = data_saved + 1
            train_set = output # it'll get reset up above
            data_created = 0

print train_set.shape
print train_labels.shape
hkl.dump(train_set, 'preinception_data/train_set-' + str(data_saved) + '.hkl', mode='w', compression='gzip', compression_opts=9)
hkl.dump(train_labels,'final_training_data/train_labels-' + str(data_saved) + '.hkl', mode='w', compression='gzip', compression_opts=9)
