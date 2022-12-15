# cap=cv2.VideoCapture('combined2.mp4')
# cap=cv2.VideoCapture(1)
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


json_file = open('model/fighting.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("fighting.h5")

print("Loaded model from disk")

cap = cv2.VideoCapture(r"C:\\Users\Aditya\Videos\Captures\biomed - sitting - PC, Mac & Linux Standalone - Unity 2020.3.25f1_ DX11 2022-11-24 19-59-00.mp4")
buffer = []
result = cv2.VideoWriter('filename.mp4', cv2.VideoWriter_fourcc(*'XVID'), 15, (1280, 646))
RRR = [{},{}, {}, {}]
leg = []
temp = []
prevpoint = 0
out = ""
framecount = 0
with mp_pose.Pose(min_detection_confidence=0.1, min_tracking_confidence=0.5) as pose:
    while (True):
        framecount += 1
        ret, frame = cap.read()
        keypoints = []
        if (not ret):
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #   image=cv2.rotate(image,cv2.ROTATE_180)
        image.flags.writeable = False
        results = pose.process(image)
        # try:
        landmarks = results.pose_landmarks.landmark
        for points in range(len(landmarks)):
            # keypoints.append(to_pixel_coords(image,[landmarks[points].x,landmarks[points].y]))
            keypoints.append([landmarks[points].x, landmarks[points].y])
        buffer.append(keypoints)

        if (len(np.array(buffer)) >= 15):  # and framecount%3==0):
            mean_val = np.mean(r.predict(np.array(buffer)), axis=0)
            res = np.argmax(mean_val)

            out = cls[res]
            buffer.pop(0)
            x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x
            y_ = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            val = ((x + y_) + (x * y_)) / 2
            temp.append(val)
            if ((out == 'walking' or out == 'standing') and len(temp) > 15):
                temp.pop()
                if (abs(temp[0] - temp[len(temp) - 1]) > 0.04):
                    out = "walking"
                else:
                    out = "standing"

            for val, actions in enumerate(mean_val):
                image = cv2.putText(image, cls[val] + ": " + str(round(100 * actions, 2)), (50, val * 70 + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
        if (len(np.array(buffer)) >= 15):
            for val, actions in enumerate(mean_val):
                image = cv2.putText(image, cls[val] + ": " + str(round(100 * actions, 2)), (50, val * 70 + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.putText(image, out, (1100, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 5), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        result.write(image)
        cv2.imshow('Mediapipes', image)
        k = cv2.waitKey(1)
        # if(k==ord('c')):
        leg.append(
            (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y))

        if k == ord('q'):
            break
    #         except:
    #             print("Exception")
    #             continue

    cap.release()
    cv2.destroyAllWindows()