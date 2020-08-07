#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 10:12:37 2020

@author: soulh
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:10:26 2020

@author: soulh
"""
import cv2
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("./trainer/trainer.yml")
id = 0
while(True):
    ret,img=cam.read()
    #gray=cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
        id,conf= rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Anirhan"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord("q")):
        break;
cam.release()
cv2.destroyAllWindows()
        