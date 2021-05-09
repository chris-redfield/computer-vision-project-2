#!/usr/bin/env python
# coding: utf-8

# ### Import libs

# In[1]:


import json
import pandas as pd
import cv2
import numpy as np
import argparse


# In[2]:


import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"


# ### Extract face frames from video files

# In[3]:


face_cascade = cv2.CascadeClassifier('models/haarcascades/haarcascade_frontalface_alt.xml')


def detect_faces(frame):
    frame_gray = cv2.equalizeHist(frame)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces
 
def get_faces_from_video_file(file_path, fake):

    cap = cv2.VideoCapture(file_path)
    file_name = file_path.split('/')[-1]

    frame_values = [80 ,100, 150, 200]
    #frame_values = [200]

    for frame_number in frame_values:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number);
        ret, frame = cap.read()
        
        cv2.imshow(f'img-{frame_number}', frame)
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        
#         print(faces)
#         print(len(faces))
        
        if faces is not None and len(faces) > 0:
            for (x,y,w,h) in faces:
                faceROI = frame[y:y+h,x:x+w]
                cv2.imshow(f'face-{frame_number}', faceROI)
                
            ### gets only the last face found
            #cv2.imshow(f'frame-{frame_number}', faceROI)
            #cv2.imwrite(file_path.replace(".mp4","") + f'-f{frame_number}.jpg',frame)
            cv2.imwrite(f'data/{fake}/{file_name}-f{frame_number}.jpg',faceROI)

        else:
            print(f'face not found for file {file_path}, frame {frame_number}')
            
    cv2.waitKey()


# In[4]:


data_path = 'deepfake-detection-challenge-data/'

with open(f'{data_path}metadata.json') as json_file:
    data = json.load(json_file)


# In[5]:


# real_path = 'data/real/'
# fake_path = 'data/fake/'

for key in data.keys():
    file_name = key
    fake = data[key]['label'].lower()
    print('extracting images for file',file_name,':',fake)
    get_faces_from_video_file(data_path + file_name, fake)


# ### Single file test

# In[5]:


#get_faces_from_video_file(data_path + 'agqphdxmwt.mp4','fake')


# In[ ]:




