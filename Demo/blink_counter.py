
import math
import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh

#Left eye
left_eye_open = True
left_eye_top_landmark = 386
left_eye_bot_landmark = 374
left_eye_counter = 0

#Right eye
right_eye_open = True
right_eye_top_landmark = 159
right_eye_bot_landmark = 145
right_eye_counter = 0

#To calculate distance between points
def euclideanDistance(point, point1):
    x1, y1 = point.x, point.y
    x2, y2 = point1.x, point1.y
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


cap = cv.VideoCapture("data/blink.mp4")
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
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:

        mp_drawing.draw_landmarks(
          image = image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_LEFT_EYE,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
          image = image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_RIGHT_EYE,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())
      
        if left_eye_open:
          if (euclideanDistance(face_landmarks.landmark[left_eye_top_landmark], face_landmarks.landmark[left_eye_bot_landmark])<=0.01):
            #print("LEFT EYE BLINKED")
            left_eye_open = False
            left_eye_counter+=1
        else:
          if (euclideanDistance(face_landmarks.landmark[left_eye_top_landmark], face_landmarks.landmark[left_eye_bot_landmark])>0.01):
            left_eye_open = True

        if right_eye_open:
          if (euclideanDistance(face_landmarks.landmark[right_eye_top_landmark], face_landmarks.landmark[right_eye_bot_landmark])<=0.01):
            #print("RIGHT EYE BLINKED")
            right_eye_open = False
            right_eye_counter+=1
        else:
          if (euclideanDistance(face_landmarks.landmark[right_eye_top_landmark], face_landmarks.landmark[right_eye_bot_landmark])>0.01):
            right_eye_open = True
            
    image = cv.resize(image, (500, 500), fx=1, fy=1, interpolation=cv.INTER_CUBIC)
    cv.putText(image, "Left eye: " + str(left_eye_counter), (10,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.putText(image, "Right eye: " + str(right_eye_counter), (10,100), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    
    cv.imshow('Demo Blinking counter', image)
    if cv.waitKey(1) == ord('q'):
      break

cap.release()
cv.destroyAllWindows