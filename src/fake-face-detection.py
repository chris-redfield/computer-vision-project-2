import cv2 
import argparse
import numpy as np
from fastai.vision.all import *
from PIL import Image
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

## Cascade ref:
## https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

## FastAI ref:
## https://github.com/fastai/fastbook/blob/master/02_production.ipynb

def is_obama(x): 
    return 'obama' in x

global learn
learn = pickle.load(open('models/svm.pkl', 'rb'))

global img1
global img2

img1 = cv2.namedWindow("img1", cv2.WINDOW_NORMAL)
img2 = cv2.namedWindow("img2", cv2.WINDOW_NORMAL)


def detect_faces(frame):
    frame_gray = cv2.equalizeHist(frame)
    faces = face_cascade.detectMultiScale(frame_gray)
    return faces

def classify(faces, frame):
    faceROI = None
    face_class = None
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        
        #bounding box dos rostos
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]

    
    if(faceROI is not None):
        cv2.imshow('img2',faceROI)
        
        # p, tensor, probs = learn.predict(t)

        # if(str(p) == 'False'):
        #     face_class = 'chris'
        # else:
        #     face_class = 'obama'

        # frame = cv2.putText(frame, face_class,(x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

        #print(face_class)

        cv2.imshow('img1', frame)


# ### Extract face frames from video files

face_cascade = cv2.CascadeClassifier('models/haarcascades/haarcascade_frontalface_alt.xml')

file_path = 'data/bill-harder-tom-cruise-deep-fake.mp4'

# Captura o video
vid = cv2.VideoCapture(file_path) 

while(True): 
      
    # Captura cada frame do video
    ret, frame = vid.read() 

    # transforma em cinza para facilitar a busca pelo  padrão
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Acha os rostos
    faces = detect_faces(gray)

    #mostra bounding boxes
    classify(faces,frame)

    # Mostra o frame atual, pode ou não estar com as bordas coloridas
    #cv2.imshow('img1', frame) 

    # Botão q para iniciar a calibração e depois sair do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Solta o objeto de captura
vid.release() 

# Destroi as janelas
cv2.destroyAllWindows() 
