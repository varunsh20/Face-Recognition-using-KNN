#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np


file_name = input("Enter your name: ")
data_path = "G:/"
skip = 0
face_data = []


cap = cv2.VideoCapture(0)

# loop runs if capturing has been initialized. 
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("C:/Users/Varun/Downloads/haarcascade_frontalface_alt (1).xml")
    eye_cascade = cv2.CascadeClassifier("C:/Users/Varun/Downloads/haarcascade_eye.xml")
    
    
    if ret==False:
        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key=lambda f: f[2]*f[3])
    
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,155),2)
        roi_color = frame[y:y+h, x:x+w] 
  
        eyes = eye_cascade.detectMultiScale(roi_color)  
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
        
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
 
    cv2.imshow("Frame",frame)
    cv2.imshow("Face Section",face_section)
    
    keypressed = cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(data_path+file_name+'.npy',face_data)
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




