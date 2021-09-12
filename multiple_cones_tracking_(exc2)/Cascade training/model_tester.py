import numpy as np
import cv2

# this is the cascade we just made. Call what you want
cascade = cv2.CascadeClassifier("./trained_cascades/3/cascade.xml")

cap = cv2.VideoCapture('../KTH FSD Perception- Intro/office.mp4')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (300,150))

    # image, reject levels level weights.
    cones = cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=15, minSize=(10,10), maxSize=(80,80))

    # add this
    for (x, y, w, h) in cones:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()