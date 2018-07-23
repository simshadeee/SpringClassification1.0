import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

broken, good = 0 ,0
for filename in glob.iglob('TruckSpring/training/broken/*.png'):
    #Crop
    # img = cv2.imread("pics/training_broken_220.png")
    img = cv2.imread(filename)
    crop_img = img[40:290, 0:150]
    cv2.imwrite("cropped.jpg", crop_img)


    #Blur
    img = cv2.imread('cropped.jpg',0)
    blur = cv2.blur(img,(200,20))
    cv2.imwrite("blurred.jpg", blur)

    # Theshold
    img = cv2.imread('blurred.jpg',0)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    cv2.imwrite("postThresh4.jpg", thresh4)

    # Theshold for black and white
    img = cv2.imread('postThresh4.jpg',0)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite("postThresh2.jpg", thresh2)


    #Largest Contour
    im2, contours, hierarchy = cv2.findContours(cv2.bitwise_not(thresh2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0.0

    for cnt in contours:
        max_area = max(cv2.contourArea(cnt), max_area)

    if max_area >= 13500.0 or max_area <= 6300.0:
        broken +=1
        print(filename, max_area)
    else:
        good +=1


    #print(max_area)

print("broken", broken)
print("good", good)
print(broken/(good * 1.0))
print(good/(broken * 1.0))
