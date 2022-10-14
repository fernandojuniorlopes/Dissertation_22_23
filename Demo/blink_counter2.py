#references used
#https://medium.com/@aiphile/eyes-blink-detector-and-counter-mediapipe-a66254eb002c

from asyncio.windows_events import NULL
import math
import cv2 as cv
import mediapipe as mp
import numpy as np
import utils
import time
import sys

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_holistic = mp.solutions.mediapipe.python.solutions.holistic

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

#Value that work best with webcam and video
eyes_closed_threshold = 6.6
fingers_threshold = 8

frame_counter = 0
if len(sys.argv) == 1:
  video = 0
else:
  video = sys.argv[1]

#example vid
example1 = "data/blink.mp4"

# landmark detection function 
def landmarksDetection(img_width, img_height, results):
    coords = [(int(point.x * img_width), int(point.y * img_height)) for point in results.landmark]
    # returning the list of tuples for each landmark
    return coords

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

with mp_holistic.Holistic(
  refine_face_landmarks=True,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5) as face_mesh:
  start_time = time.time()

  while cap.isOpened():
    frame_counter+=1
    
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

    if results.face_landmarks:

      #Normalization of the landmarks
      face_coords = landmarksDetection(width, height, results.face_landmarks)
      
      #Drawing the landmarks
      cv.polylines(image,  [np.array([face_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,0,355), 1, cv.LINE_AA)
      cv.polylines(image,  [np.array([face_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,355,0), 1, cv.LINE_AA)

      #Calculating distance between top and bottom of the eye
      distance_landmarks_left = euclideanDistance(face_coords[left_eye_upper_landmark], face_coords[left_eye_lower_landmark])
      distance_landmarks_right = euclideanDistance(face_coords[right_eye_upper_landmark], face_coords[right_eye_lower_landmark])
      
      #State machine that counts the amount of times each eye closes
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
    
    if results.left_hand_landmarks:
      left_hand_coords = landmarksDetection(width, height, results.left_hand_landmarks)
      
      mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_hand_connections_style()) 

      if euclideanDistance(left_hand_coords[8], left_hand_coords[4]) <= fingers_threshold:
        left_eye_counter = 0
    
    if results.right_hand_landmarks:
      right_hand_coords = landmarksDetection(width, height, results.right_hand_landmarks)
      mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_hand_connections_style()) 

      if euclideanDistance(right_hand_coords[8], right_hand_coords[4]) <= fingers_threshold:
        right_eye_counter = 0

    end_time = time.time() - start_time

    fps = round(frame_counter/end_time,1)
    
    cv.putText(image, "FPS: " + str(fps), (10, height-50), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 127, 0), 2)
    cv.putText(image, "Left eye: " + str(left_eye_counter), (10,50), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
    cv.putText(image, "Right eye: " + str(right_eye_counter), (10,100), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    
    cv.imshow('Demo Blinking counter', image)
    if cv.waitKey(1) == ord('q'):
      break

cap.release()
cv.destroyAllWindows