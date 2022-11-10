#references used
#https://medium.com/@aiphile/eyes-blink-detector-and-counter-mediapipe-a66254eb002c

import csv
import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import utils
import time
import sys

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh

#Left eye
left_eye_open = True
left_eye_upper_landmark = 386
left_eye_lower_landmark = 374
left_eye_counter = 0
eyeOpenness_left=0

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_UPPER_EYE = [398, 384, 385, 386, 387, 388, 466]
LEFT_LOWER_EYE = [382, 381, 380, 374, 373, 390, 249]
LEFT_CENTER_EYE = [362, 263]

#Right eye
right_eye_open = True
right_eye_upper_landmark = 159
right_eye_lower_landmark = 145
right_eye_counter = 0
eyeOpenness_right = 0


RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
RIGHT_UPPER_EYE = [173, 157, 158, 159, 160, 161, 246]
RIGHT_LOWER_EYE = [155, 154, 153, 145, 144, 163, 7]
RIGHT_CENTER_EYE = [133, 33]

#Value that work best with webcam and video
eyes_closed_threshold = 0.5
pic1 = False
pic2 = False

frame_counter = 0
if len(sys.argv) == 1:
  video = 0
else:
  video = sys.argv[1]

#example vid
example1 = "data/blink.mp4"

def face_align(mesh_coords, image):
  #Percentage to control how much of the face is visible after alignment(the smaller the percentage the more zoomed in the face is)
  desired_left_eye = (0.3, 0.3)
  
  #Face width (default 256px)
  desired_face_width = 256

  #Face height (default 256px)
  desired_face_height = 256 

  #Coordinates of the landmarks of each eye
  left_eye_coords = np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)
  right_eye_coords = np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)

  #Computing the center of each eye
  left_eye_center = left_eye_coords.mean(axis=0).astype("int32")
  right_eye_center = right_eye_coords.mean(axis=0).astype("int32")

  #Computing the angle between the eyes center
  dY = right_eye_center[1] - left_eye_center[1]
  dX = right_eye_center[0] - left_eye_center[0]
  angle = np.degrees(np.arctan2(dY, dX)) - 180

  #computing the desired right eye x coordinate based on the x coordinte of the left eye
  desired_right_eye_x = 1.0 - desired_left_eye[0]

  #computing euclidian distance between eyes
  distance_eyes = np.sqrt((dX ** 2) + (dY **2))

  #computing desired distance between eyes
  desired_distance = (desired_right_eye_x - desired_left_eye[0])

  #Scaling eye distance based on the desired width
  desired_distance *= desired_face_width

  #computing scale dividing desired_distance by the eye distance
  scale = desired_distance / distance_eyes
  
  #Computing the coordinates of the center point between the eyes
  eyes_center = (int((left_eye_center[0] + right_eye_center[0])//2), int((left_eye_center[1] + right_eye_center[1])//2))

  #apply rotation and scale to the matrix
  M = cv.getRotationMatrix2D(eyes_center, angle, scale)

  #update the translation component of the matrix
  tX = desired_face_width * 0.5
  tY = desired_face_height * desired_left_eye[1]
  M[0, 2] += (tX - eyes_center[0])
  M[1, 2] += (tY - eyes_center[1])

  #apply the affine transformation
  (w, h) = (desired_face_width, desired_face_height)
  output = cv.warpAffine(image, M, (w, h), flags = cv.INTER_CUBIC)
  
  return output

# landmark detection function 
def landmarksDetection(img_width, img_height, results):
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    
    # returning the list of tuples for each landmark
    return mesh_coord

#To calculate distance between points
def euclideanDistance(point, point1):
    x1, y1 = point
    x2, y2 = point1
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

cap = cv.VideoCapture(video)

if not cap.isOpened():
  print("Cannot open camera")
  exit()

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

with mp_face_mesh.FaceMesh(max_num_faces = 1,
  refine_landmarks=True,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5) as face_mesh:
  start_time = time.time()

  while cap.isOpened():
    frame_counter+=1
    sum_left_upper_eye_y = 0.0
    sum_left_lower_eye_y = 0.0
    sum_right_upper_eye_y = 0.0
    sum_right_lower_eye_y = 0.0
    
    recieved, image = cap.read()

    if not recieved:
      print("Can't receive frame (stream end?). Existing ...")
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    # height, width = image.shape[:2]

    if results.multi_face_landmarks:

      #Normalization of the landmarks
      mesh_coords = landmarksDetection(width, height, results)
      
      #Drawing the landmarks
      cv.polylines(image,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,0,355), 1, cv.LINE_AA)
      cv.polylines(image,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,355,0), 1, cv.LINE_AA)
      
      image = face_align(mesh_coords,image)
    
    end_time = time.time() - start_time
    fps = round(frame_counter/end_time,1)
    
    # cv.putText(image, "FPS: " + str(fps), (10, height-50), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 127, 0), 2)
    cv.imshow('Demo Blinking counter', image)
    
    if cv.waitKey(1) == ord('q'):
      break

# close the file
f.close()
cap.release()
cv.destroyAllWindows