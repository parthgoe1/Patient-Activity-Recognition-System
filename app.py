from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
import os
import time
import threading
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
import base64
import pandas as pd
from keras.models import load_model
import math
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import json
from twilio.rest import Client

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'flv', 'mov', 'avi', 'mp4'}
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
app.config['JSON_SORT_KEYS'] = False

cls = ['falling', 'pain/cough', 'sitting', 'sleeping', 'standing', 'walking']

final_output = []

df = pd.DataFrame(columns=['frame_id', 'activity', 'percentage'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def distance(p1,p2):
    d = math.sqrt(math.pow(p1[0] - p2[0], 2) +math.pow(p1[1] - p2[1], 2) +math.pow(p1[2] - p2[2], 2)* 1.0)
    return d

def features(landmarks):
    feat=[]
    landmarks=list(landmarks)
    #1st:15-16
    rwrist=[landmarks[16].x,landmarks[16].y,landmarks[16].z]
    lwrist=[landmarks[15].x,landmarks[15].y,landmarks[15].z]
    feat.append(distance(rwrist,lwrist))
                
    #2nd: 15-11
    lshoulder= [landmarks[11].x,landmarks[11].y,landmarks[11].z]  
    feat.append(distance(rwrist,lshoulder))
                
    #3rd: 12-16
    rshoulder = [landmarks[12].x,landmarks[12].y,landmarks[12].z] 
    feat.append(distance(rshoulder,rwrist))
                
    #4th: 15-23
    lhip= [landmarks[23].x,landmarks[23].y,landmarks[23].z]
    feat.append(distance(lhip,lwrist))
                
    #5th: 16-24
    rhip= [landmarks[24].x,landmarks[24].y,landmarks[24].z] 
    feat.append(distance(rhip,lwrist))
                
    #6th: 23-24
    feat.append(distance(lhip,rhip))
    
    #7th: 16-28
    rankle=[landmarks[28].x,landmarks[28].y,landmarks[28].z]
    feat.append(distance(rankle,rwrist))
                
    #8th: 15-27
    lankle=[landmarks[27].x,landmarks[27].y,landmarks[27].z]
    feat.append(distance(lankle,lwrist))
    
    #9th: 23-27
    feat.append(distance(lhip,lankle))
                
    #10th: 24-28
    feat.append(distance(rhip,rankle))
                
    #11th: 28-27
    feat.append(distance(lankle,rankle))
                
    #12th: 5-30
    rheel=[landmarks[30].x,landmarks[30].y,landmarks[30].z]
    reye=[landmarks[5].x,landmarks[5].y,landmarks[5].z]
    feat.append(distance(reye,rheel))
                
    #13th: 2-29
    leye=[landmarks[2].x,landmarks[2].y,landmarks[2].z]
    lheel=[landmarks[29].x,landmarks[29].y,landmarks[29].z]
    feat.append(distance(leye,lheel))
    
    #14th: 14-27
    relbow=[landmarks[14].x,landmarks[14].y,landmarks[14].z]
    feat.append(distance(relbow,lankle))
                
    #15th: 13-28
    lelbow=[landmarks[13].x,landmarks[13].y,landmarks[13].z]
    feat.append(distance(lelbow,rankle))
    
    #16th: 0-15
    nose=[landmarks[0].x,landmarks[0].y,landmarks[0].z]
    feat.append(distance(nose,lwrist))
    
    #17th: 0-16
    feat.append(distance(nose,rwrist))
    
    return feat


@app.route("/upload-video", methods=["POST"])
def upload_video():
    global df
    df = pd.DataFrame(columns=['frame_id', 'activity', 'percentage'])
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        thread = threading.Thread(target=process_video, kwargs={
            'file_path': file_path})
        thread.start()

        resp = jsonify({'message': 'File successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify({'message': 'Allowed file types are flv, mov, avi, mp4'})
        resp.status_code = 400
        return resp


@app.route("/get-current-activity", methods=["GET"])
def get_current_activity():
    global final_output
    if len(final_output) == 0:
        return jsonify([])
    results = []
    for x in range(60):
        if len(final_output) == 0:
            break
        results.append(final_output.pop(0))
    return results


@app.route("/get-report", methods=["GET"])
def get_report():
    # df.to_csv('res.csv', header=True, index=False)
    x = df.groupby(['frame_id'])['percentage'].transform(max) == df['percentage']
    x = df[x].groupby('activity').count().reset_index()
    x['time'] = round(x['frame_id'] / 28)
    x_di = x[['activity', 'time']].to_dict('records')
    di_res = {x:0 for x in cls}
    for rec in x_di:
        di_res[rec['activity']] = rec['time']
    return {"report": di_res}

def to_pixel_coords(img,relative_coords):
    SCREEN_DIMENSIONS = (img.shape[1],img.shape[0])
    return tuple(round(coord * dimension) for coord, dimension in zip(relative_coords, SCREEN_DIMENSIONS))

def process_video(**kwargs):
    global call
    n_features = 116
    n_steps = 2
    n_length = 25
    file_name = kwargs.get('file_path', {})
    global final_output, df
    cap = cv2.VideoCapture(file_name)
    buffer = []
    temp = []
    framecount = 0
    painframe = 0
    with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.5) as pose:
        prev_frame_time = 0
        while (True):
            new_frame_time = time.time()
            framecount += 1
            ret, frame = cap.read()
            keypoints = []
            if (not ret):
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            # converting the fps into integer
            fps = int(fps)

            try:
                landmarks = results.pose_landmarks.landmark
                for points in range(len(landmarks)):
                    keypoints.extend([landmarks[points].x, landmarks[points].y, landmarks[points].z])
                keypoints.extend(features(landmarks))
                buffer.append(keypoints)
                if (len(np.array(buffer)) >= 50):  # and framecount%3==0):
                    buffer_array = np.array(buffer).reshape((50, n_features))
                    # buffer_array_scaled = sc.transform(buffer_array)
                    buffer_array_scaled = buffer_array
                    buffer_array_scaled = buffer_array_scaled.reshape((1, n_steps, n_length, n_features))
                    predicted_activity = model.predict(buffer_array_scaled)
                    results_new = np.argmax(predicted_activity, axis = 1)
                    out=cls[results_new[0]]
                    if out in ['falling', 'pain/cough']:
                        painframe+=1
                        if painframe > 40 and call == False:
                            print('Call executed')
                            called = client.calls.create(
                                            twiml='<Response><Say>Patient in room number 111 needs assistance. Please attend to the patient immediately.</Say></Response>',
                                            from_='+1 562 573 7381',
                                            to='+17327637131'
                            )
                            call = True
                    else:
                        painframe = 0
                    # results_new = np.argmax(predicted_activity, axis = 1)
                    current_activity = {cls[i]:round(predicted_activity[0][i]*100,2) for i in range(len(cls)) }

                    current_activity = {k: v for k, v in sorted(current_activity.items(), key=lambda item: item[1], reverse=True)}
                    for i in range(len(cls)):
                        df = pd.concat([df, pd.DataFrame({'frame_id': [framecount], 'activity': [cls[i]], 'percentage': [round(predicted_activity[0][i]*100,2)]})])
                    buffer.pop(0)

                    # x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
                    # y_ = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                    # val = ((x + y_) + (x * y_)) / 2
                    # temp.append(val)
                    # max_key = max(current_activity, key=current_activity.get)
                    # image = cv2.putText(image, max_key+ ":" + str(current_activity[max_key]), (50, val * 70 + 100),
                    #                     cv2.FONT_HERSHEY_SIMPLEX,
                    #                     1, (255, 0, 0), 2, cv2.LINE_AA)
                    # current_activity = {k: v for k, v in sorted(current_activity.items(), key=lambda item: item[1], reverse=True)}
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                              )

                    
                    
                    
                    retval, bf = cv2.imencode('.jpg', image)
                    data_base64 = base64.b64encode(bf)  # encode to base64 (bytes)
                    data_base64 = data_base64.decode()  # convert bytes to string
                    current_data = {
                        "current_frame": 'data:image/jpeg;base64,' + data_base64,
                        "current_activity": current_activity
                    }
                    final_output.append(current_data)
            except Exception as e:
                print(e)
                continue
        cap.release()
    print('Finished Process')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    account_sid="AC8bd91209dc5dcfbdc8ebc429b6635a79"
    auth_token="8bf6d0a0873e0444499e20edd8fb4d38"
    client = Client(account_sid,auth_token)
    call = False
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    n_outputs = 6
    n_length = 25
    n_features = 116
    # Initialising the RNN
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units = 100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units = 100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units = 100, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units = 100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))

    # Loading model
    model = load_model('./model/my_model50_2.h5')
    # app.run(debug=True, host="0.0.0.0", port=8888)
    app.run(debug=True, port=8887)
