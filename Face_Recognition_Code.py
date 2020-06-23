#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import os


cap = cv2.VideoCapture(0) #Captures frames from webcam.

data_path = "G:/"
class_id = 0
face_data = []
labels = []
names = {}

#Here we define the KNN algorithm for classification

def distance(x1,x2):  #Measures euclidean distance between two point.
    return np.sqrt(sum((x1-x2)**2))

def knn(train,test,k=5): #KNN alogorithm.
    dist = []
    for i in range(train.shape[0]):
        ix = train[i,:-1]
        iy = train[i,-1]
        d = distance(test,ix)
        dist.append([d,iy])
    dk = sorted(dist,key = lambda x: x[0])[:k] 
    label = np.array(dk)[:,-1]
    output = np.unique(label,return_counts = True)
    index = np.argmax(output[1])
    classes = output[0][index]
    return classes
    

for file in os.listdir(data_path):
    if file.endswith('.npy'):
        names[class_id] = file[:-4]
        data_item = np.load(data_path+file)
        face_data.append(data_item)
        target = class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
    
face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))



training_set = np.concatenate((face_dataset,face_labels),axis = 1)

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("C:/Users/Varun/Downloads/haarcascade_frontalface_alt (1).xml")#Creates a classifier object.
    eye_cascade = cv2.CascadeClassifier("C:/Users/Varun/Downloads/haarcascade_eye.xml")
    if ret==False:
        continue
    faces = face_cascade.detectMultiScale(frame,1.3,5) #Detects face in the frame.
    
    for face in faces[-1:]:
        x,y,w,h = face
        
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset] #Here instead of the entire region we have sectioned out the target region ie the face
        face_section = cv2.resize(face_section,(100,100)) 
        
        out = knn(training_set,face_section.flatten())
        pred_output = names[int(out)] #Here we assign the name of the person to pred_output
        
        roi_color = frame[y:y+h, x:x+w] 
  
        eyes = eye_cascade.detectMultiScale(roi_color)  
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
        
        
        
        cv2.putText(frame,pred_output,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,120,100),2,cv2.LINE_AA) #Dsiplay the name of the person
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,155),2) #Creates a rectangle around the face.


    cv2.imshow("Frame",frame)
    
    keypressed = cv2.waitKey(1) & 0xFF
    if keypressed==ord('q'): # If we press q it closes the window
        break

cap.release()
cv2.destroyAllWindows()        






# In[ ]:





# In[ ]:




