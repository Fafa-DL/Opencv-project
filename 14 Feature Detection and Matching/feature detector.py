import cv2
import numpy as np
 
img1 = cv2.imread('ImagesQuery/XBOX Kinect Sports.jpg',0)
#img2 = cv2.imread('ImagesTrain/Kinect.jpg',0)
img2 = cv2.imread('ImagesTrain/Last.jpg',0)
 
orb = cv2.ORB_create(nfeatures=1000)
 
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# imgKp1 = cv2.drawKeypoints(img1,kp1,None)
# imgKp2 = cv2.drawKeypoints(img2,kp2,None)
 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
 
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
 
# cv2.imshow('Kp1',imgKp1)
# cv2.imshow('Kp2',imgKp2)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.waitKey(0)