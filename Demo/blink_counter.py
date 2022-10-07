#references used
#https://medium.com/@aiphile/eyes-blink-detector-and-counter-mediapipe-a66254eb002c

import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import utils

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh

#Left eye
left_eye_open = True
left_eye_upper_landmark = 386
left_eye_lower_landmark = 374
left_eye_counter = 0

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

#Right eye
right_eye_open = True
right_eye_upper_landmark = 159
right_eye_lower_landmark = 145
right_eye_counter = 0

eyes_closed_threshold = 6.7

#example vid
example1 = "data/blink.mp4"

# landmark detection function 
def landmarksDetection(img, results):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

    # returning the list of tuples for each landmarks 
    return mesh_coord

#To calculate distance between points
def euclideanDistance(point, point1):
    x1, y1 = point
    x2, y2 = point1
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

cap = cv.VideoCapture(example1)

if not cap.isOpened():
  print("Cannot open camera")
  exit()

with mp_face_mesh.FaceMesh(max_num_faces = 1,
  refine_landmarks=True,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5) as face_mesh:

  while cap.isOpened():
    ret, image = cap.read()

    if not ret:
      print("Can't receive frame (stream end?). Existing ...")
      break

    image.flags.writeable = True
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
      mesh_coords = landmarksDetection(image, results)
      cv.polylines(image,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,0,355), 1, cv.LINE_AA)
      cv.polylines(image,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,355,0), 1, cv.LINE_AA)

      distance_landmarks_left = euclideanDistance(mesh_coords[left_eye_upper_landmark], mesh_coords[left_eye_lower_landmark])
      distance_landmarks_right = euclideanDistance(mesh_coords[right_eye_upper_landmark], mesh_coords[right_eye_lower_landmark])

      if left_eye_open:
        if distance_landmarks_left<=eyes_closed_threshold:
          left_eye_open = False
          left_eye_counter+=1
      else:
        if distance_landmarks_left>eyes_closed_threshold:
          left_eye_open = True

      if right_eye_open:
        if distance_landmarks_right<=eyes_closed_threshold:
          right_eye_open = False
          right_eye_counter+=1
      else:
        if distance_landmarks_right>eyes_closed_threshold:
          right_eye_open = True

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)/2)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)/2)

    image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)

    cv.putText(image, "Left eye: " + str(left_eye_counter), (10,50), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
    cv.putText(image, "Right eye: " + str(right_eye_counter), (10,100), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    
    cv.imshow('Demo Blinking counter', image)
    if cv.waitKey(1) == ord('q'):
      break

cap.release()
cv.destroyAllWindows