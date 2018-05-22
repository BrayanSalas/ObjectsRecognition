import cv2
import time 
#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('C:/Users/braya/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
test1 = cv2.imread('foto1.jpg')
#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
#let's detect multiscale (some images may be closer to camera than others) images 
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in faces:     
         cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('asd', test1)  
cv2.waitKey(0) 
cv2.destroyAllWindows()
