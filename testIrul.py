import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import winsound

IMG_SIZE = (64,56)
B_SIZE = (34, 26)
margin = 95
class_labels = ['center','left', 'right'] 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

font_letter = cv2.FONT_HERSHEY_PLAIN
model = load_model('models/gazev3.1.h5')
model_b = load_model('models/blinkdetection.h5')


def detect_gaze(eye_img):
    pred_l = model.predict(eye_img)
    accuracy = int(np.array(pred_l).max() * 100)
    gaze = class_labels[np.argmax(pred_l)]
    return gaze


def detect_blink(eye_img):
    pred_B = model_b.predict(eye_img)
    status = pred_B[0][0]
    status = status*100
    status = round(status,3)
    return  status

   
def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect
 
# pattern = []
# frames = 10
# pattern_length = 0 
import os

# Path to the directory containing the videos
video_directory = r'C:\Users\23020235\Desktop\uploads'
a=[]
# Loop through all files in the directory
for filename in os.listdir(video_directory):

    if filename.endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv')):
        video_path = os.path.join(video_directory, filename)

 
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {filename}")
        else:
            frames_to_blink = 6
            blinking_frames = 0
            count=0
            # Specify the path to the uploaded video file
            # video_path = 'hp.mp4'  
            video_capture = cv2.VideoCapture(video_path)
            while True:
                output = np.zeros((900,820,3), dtype="uint8")
                ret, img = video_capture.read()
                img = cv2.flip(img,flipCode = 1)
                h,w = (112,128)	
                if not ret:
                    break

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                

                faces = detector(gray)

                for face in faces:
                    shapes = predictor(gray, face)
                    

                    for n in range(36,42):
                        x= shapes.part(n).x
                        y = shapes.part(n).y
                        next_point = n+1
                        if n==41:
                            next_point = 36 
                        
                        x2 = shapes.part(next_point).x
                        y2 = shapes.part(next_point).y

                    for n in range(42,48):
                        x= shapes.part(n).x
                        y = shapes.part(n).y
                        next_point = n+1
                        if n==47:
                            next_point = 42 
                        
                        x2 = shapes.part(next_point).x
                        y2 = shapes.part(next_point).y 
                    shapes = face_utils.shape_to_np(shapes)
                    #~~~~~~~~~~~~~~~~~56,64 EYE IMAGE~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
                    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48]) 
                    #~~~~~~~~~~~~~~~~~FOR THE BLINK DETECTION~~~~~~~~~~~~~~~~~~~~~~~
                    eye_blink_left = cv2.resize(eye_img_l.copy(), B_SIZE)
                    eye_blink_right = cv2.resize(eye_img_r.copy(), B_SIZE)
                    eye_blink_left_i = eye_blink_left.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255.
                    eye_blink_right_i = eye_blink_right.reshape((1, B_SIZE[1], B_SIZE[0], 1)).astype(np.float32) / 255. 
                    #~~~~~~~~~~~~~~~~~~PREDICTION PROCESS~~~~~~~~~~~~~~~~~~~~~~~~~~#
                    
                    status_l = detect_blink(eye_blink_left_i)  
                    status_r = detect_blink(eye_blink_right_i)  
                    #~~~~~~~~~~~~~~~~~~~~~~~FINAL_WINDOWS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                    print(status_l)
                    print(status_r)
                    if status_l < 10 and status_r < 10: 
                        print("blinking")
                        count+=1
                    else:
                        print("ga ada blink")
            a.append("jumlah Blink :" + str(count)+" divideo :" +video_path)
print(a)

 

