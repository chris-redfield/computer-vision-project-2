#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
import pandas as pd
import cv2
import numpy as np


# In[2]:


data_path = 'deepfake-detection-challenge-data/'
with open(f'{data_path}metadata.json') as json_file:
    data = json.load(json_file)


# In[3]:


real_path = 'data/real/'
fake_path = 'data/fake/'

for key in data.keys():
    print(key)
    print(data[key])


# In[16]:


# # Captura o video
# vid = cv2.VideoCapture(data_path + 'slwkmefgde.mp4') 
# cap.get(cv2.CAP_PROP_POS_FRAMES)
# while not vid.isOpened():
#     vid = cv2.VideoCapture(data_path + 'slwkmefgde.mp4')
#     cv2.waitKey(1000)
#     print("Wait for the header")

    
# while(True):
#     # Captura cada frame do video
#     ret, frame = vid.read() 


# In[15]:


cap = cv2.VideoCapture(data_path + 'slwkmefgde.mp4')
count = 0
cap.get(cv2.CAP_PROP_POS_FRAMES)
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name', frame)
    cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('teste')


# In[11]:


data_path + 'slwkmefgde.mp4'


# In[14]:


cv2.__version__


# In[ ]:




