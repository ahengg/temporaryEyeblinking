import cv2, dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model 
from fastapi import FastAPI, File, UploadFile
import os 
import tensorflow as tf
import psycopg2
import cv2 
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  
from keras.models import load_model
import random
from PIL import Image, ImageChops, ImageEnhance 

import os
import glob
from pylab import imshow
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch 
import pandas as pd
import albumentations as albu

from albumentations.pytorch.transforms import ToTensorV2
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping

from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping
modelEff = create_model("tf_efficientnet_b3_ns")
modelEff.eval() 
folder_path = r'C:\Users\23020235\FakeImageDetector\TESTING TERAKHIR\REAL LOSS' 
model_LSTM = load_model(r'/root/Documents/api_liveness_blink_dev/v2/models/model5kKantor5kInet7030LSTM.h5')
# def convert_to_ela_image(path, quality):
#     im = Image.open(path).convert('RGB')

#     # Save the original image to a BytesIO buffer instead of a file
#     buffer = BytesIO()
#     im.save(buffer, 'JPEG', quality=quality)

#     # Open the resaved image directly from the buffer
#     buffer.seek(0)
#     resaved_im = Image.open(buffer)

#     ela_im = ImageChops.difference(im, resaved_im)

#     extrema = ela_im.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff

#     ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
#     return ela_im

#
# Replace these values with your actual database connection details
db_params = {
    'host': 'localhost',
    'database': 'liveness_memo',
    'user': 'postgres',
    'password': 'test123',
    'port': '8000'
}

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(**db_params)

# Create a cursor object to interact with the database
cursor = conn.cursor()
#
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
app = FastAPI()
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

   
def crop_eye(gray, eye_points):
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
 


@app.post("/blinkCalculate")
async def upload_video(file: UploadFile):
    count=0
    countFrame=0
    # You can process the uploaded file here
    # For example, you can save it to the server or perform any required operations
    print(file)
    # Replace the file_path with your desired file location
    file_path = f"uploads/{file.filename}"
    passed=False
    pass_value=0
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    

    num = random.uniform(0, 11000)
        

    # Specify the path to the uploaded video file
    video_path = file_path  
    video_capture = cv2.VideoCapture(video_path)

    frames_to_blink = 6
    blinking_frames = 0
    count=0
    livenessPassivePass=False
    # Specify the path to the uploaded video file
    # video_path = 'hp.mp4'  
    video_capture = cv2.VideoCapture(video_path)
    countFrames=0
    while True:
            
        output = np.zeros((900,820,3), dtype="uint8")
        ret, img = video_capture.read()
        # if(countFrames==0):
        #     frame_path = "frames.jpg"
        #     cv2.imwrite(frame_path, img)
        #     # Process the frame (e.g., convert it to grayscale)
        #     #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     # elasingle=np.array(convert_to_ela_image(frame_path, 80).resize((128, 128))).flatten() / 255.0
        #     # #elasingle=np.array(elasingle)
        #     # X = elasingle.reshape(-1, 128, 128, 3)  
            
        #     # hasil=(model_LSTM.predict(X)) 
        #     # retrunHasil=model_LSTM.predict(X)[0][1]
        #     # print(hasil) 
        #     # hasil=np.argmax(hasil)
        #     image_replay = load_rgb(frame_path)
        #     transform = albu.Compose([albu.PadIfNeeded(min_height=400, min_width=400),
        #                             albu.CenterCrop(height=400, width=400), 
        #                             albu.Normalize(p=1), 
        #                             albu.pytorch.ToTensorV2(p=1)], p=1)
        #     with torch.no_grad():
        #         prediction = modelEff(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]  
        #     print(np.argmax(prediction)) 
        #     if(np.argmax(prediction)==0):
        #         hasil=("real")
        #     elif (np.argmax(prediction)==1):
        #         hasil=("replay")
        #     elif (np.argmax(prediction)==2):
        #         hasil=("printed")
        #     else:
        #         hasil=("2dMask")
        #     if(hasil=="real"):
        #         livenessPassivePass=True
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

            if status_l < 10 and status_r < 10:  
                passed=True
                count+=1
                pass_value=1
        countFrames=+1

    # Sample data
    # nip_value = num
    # total_blink_value = count
    # tanggal_absen_value = '2023-10-10'
    # liveness_passive_value = 0
    # #pass_value = 1
    # file_name=file.filename
    # Example query: insert data into the 'liveness_score' table with parameters
    # query = """
    #     INSERT INTO liveness_score (nip, total_blink, tanggal_absen, liveness_passive, pass,file_name)
    #     VALUES (%s, %s, %s, %s, %s,%s);
    # """
    # # Use the execute method with the query and a tuple of parameter values
    # cursor.execute(query, (nip_value, total_blink_value, tanggal_absen_value, liveness_passive_value, pass_value,file_name))

    # # Commit the changes
    # conn.commit()

    # # Print a message indicating success
    # print("Data inserted successfully")

    # # Close the cursor and connection
    # cursor.close()
    # conn.close()
    # return {"jumlah Blink": count,"PassActive":passed,"PassPassive":livenessPassivePass}
    return {"jumlah Blink": count,"PassActive":passed}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)