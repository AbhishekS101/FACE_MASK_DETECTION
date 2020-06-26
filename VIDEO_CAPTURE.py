import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while cap.isOpened():

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1,4)

    for (x, y, w, h) in faces:
        # face_img = gray[y:y + w, x:x + w]
        # resized = cv2.resize(face_img, (100, 100))
        # normalized = resized / 255.0
        # reshaped = np.reshape(normalized, (1, 100, 100, 1))
        # result = 0
        #
        # label = np.argmax(result, axis=1)[0]
        label = 0

        cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', frame)
    key = cv2.waitKey(1)

    if (key == 27):
        break

cv2.destroyAllWindows()
cap.release()

