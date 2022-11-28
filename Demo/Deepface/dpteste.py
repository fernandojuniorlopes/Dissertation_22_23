#emotion_detection.py
import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process

imgpath = "Data/happy.jpg"
imgpath2 = "Data/emilia.jpg"
image = cv2.imread(imgpath)
image2 = cv2.imread(imgpath2)
imager = cv2.resize(image, (720, 720))
image2r = cv2.resize(image2, (720, 720))
analyze = DeepFace.analyze(image , enforce_detection=False, actions=['emotion'])  #here the first parameter is the image we want to analyze #the second one there is the action
print(analyze)
cv2.imshow('Image1', imager)
cv2.imshow('Image2', image2r)
verification = DeepFace.verify(imgpath, imgpath2, enforce_detection=False)
print(verification)

cv2.waitKey(0)