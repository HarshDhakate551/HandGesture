import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import  time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 30
imagesize = 300

folder = "Data_images/Z"
counter = 0

lables = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    imageOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hands = hands[0]
        x, y, w, h = hands['bbox']

        image_defined = np.ones((imagesize,imagesize,3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset , x-offset:x + w+offset ]

        imgCropShape = imgCrop.shape


        aspectRation = h / w

        if aspectRation > 1:
            k = imagesize/h
            wCal = math.ceil(k*w)
            imageResize = cv2.resize(imgCrop,(wCal,imagesize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imagesize - wCal)/2)
            image_defined[:, wGap: wCal + wGap] = imageResize
            prediction, index = classifier.getPrediction(image_defined)
            print(prediction,index)

        else:
            k = imagesize/w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imgCrop, (imagesize , hCal))
            imageResizeShape = imageResize.shape
            hGap = math.ceil((imagesize - hCal) / 2)
            image_defined[hGap: hCal + hGap, :] = imageResize
            prediction, index = classifier.getPrediction(image_defined)

        cv2.putText(imageOutput,lables[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 1)

        cv2.imshow("ImageCroped", imgCrop)
        cv2.imshow("Imagedefined",image_defined)

    cv2.imshow("Image", imageOutput)
    cv2.waitKey(1)


jkbjkbjkb
