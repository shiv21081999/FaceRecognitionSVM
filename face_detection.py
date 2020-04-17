import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    

while True:
    ret , frame = cam.read()
    
    if ret == False:
        continue
    
        
    faces = face_cascade.detectMultiScale(frame, 1.5, 5)
    
    for face in faces:
        x, y, w, h = face
        frame = cv2.rectangle(frame, (x, y), (x+w,y+h), (255,255,255), 2)
    
    cv2.imshow("Face Detection", frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
    
cam.release()
