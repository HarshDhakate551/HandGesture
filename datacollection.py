import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import  time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 30
imagesize = 300

folder = "Data_images/harsh"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hands = hands[0]
        x, y, w, h = hands['bbox']

        image_defined = np.ones((imagesize,imagesize,3), np.uint8)*255
        imgCrop = img[y-offset:y + h+offset , x-offset:x + w+offset ]

        imgCropShape = imgCrop.shape


        aspectRation = h/w

        if aspectRation>1:
            k = imagesize/h
            wCal = math.ceil(k*w)
            imageResize = cv2.resize(imgCrop,(wCal,imagesize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imagesize - wCal)/2)
            image_defined[:, wGap: wCal + wGap] = imageResize

        else:
            k = imagesize/w
            hCal = math.ceil(k * h)
            imageResize = cv2.resize(imgCrop, (imagesize , hCal))
            imageResizeShape = imageResize.shape
            hGap = math.ceil((imagesize - hCal) / 2)
            image_defined[hGap: hCal + hGap, :] = imageResize

        cv2.imshow("ImageCroped", imgCrop)
        cv2.imshow("Imagedefined",image_defined)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',image_defined)
        print(counter)

