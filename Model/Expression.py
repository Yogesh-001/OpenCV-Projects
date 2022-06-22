import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

emotions={0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

cam=cv2.VideoCapture(0)
json_file=open(r"C:\Users\mural\PycharmProjects\FaceExpression\Model\model.json","r")
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights(r"C:\Users\mural\PycharmProjects\FaceExpression\Model\model.h5")
while True:
    ret,frame = cam.read()
    if not ret:
        break
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_faces = face_cascade.detectMultiScale(gray_frame,scaleFactor=1.05, minNeighbors=5)
    for (x, y, w,h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+15), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y+h,x:x +w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction=model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotions[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()