import cv2 
import argparse
import numpy as np
import random
from fastai.vision.all import *
from PIL import Image
import radialProfile
from scipy.interpolate import griddata
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
    #feature number
    N = 300
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        
        #bounding box dos rostos
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frame[y:y+h,x:x+w]

    #print(faceROI.shape)


    if(faceROI is not None and len(faces)>0):
        # print(faceROI.shape)
        cv2.imshow('img2',faceROI)
        
        faceROI = faceROI[:,:,0]

        f = np.fft.fft2(faceROI)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

        # Calculate the azimuthally averaged 1D power spectrum
        points = np.linspace(0,N,num=psd1D.size) # coordinates of a
        xi = np.linspace(0,N,num=N) # coordinates for interpolation

        interpolated = griddata(points,psd1D,xi,method='cubic')
        interpolated /= interpolated[0]
        preds = learn.predict([interpolated])
        # print(preds)

        if(preds[0] == 0):
            face_class = 'fake'
        else:
            face_class = 'real'

        frame = cv2.putText(frame, face_class,(x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)

        #print(face_class)

        cv2.imshow('img1', frame)


# ### Extract face frames from video files

face_cascade = cv2.CascadeClassifier('models/haarcascades/haarcascade_frontalface_alt.xml')

#file_path = 'data/bill-harder-tom-cruise-deep-fake.mp4'
file_path = 'data/fake-1.mp4'

# Captura o video
vid = cv2.VideoCapture(file_path) 

while(True): 
      
    # Captura cada frame do video
    ret, frame = vid.read() 

    # transforma em cinza para facilitar a busca pelo  padrão
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    

    do_classify = [True, False, False, False]
    if(random.choice(do_classify)):
        # Acha os rostos
        faces = detect_faces(gray)
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
