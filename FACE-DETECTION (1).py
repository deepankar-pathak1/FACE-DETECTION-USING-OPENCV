#!/usr/bin/env python
# coding: utf-8

# In[10]:


# IMPORT BASIC LIB.
import cv2
import numpy as np


# In[11]:


# create Dataset
def Create_Dataset(): 
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)  #CAPTURING IMAGE
    count = 0
    while True:
        ret,image = cam.read()
        if ret == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,10)
            for x,y,w,h in faces:
                count = count + 1
                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
                roi_face = image[y:y+h, x:x+w] ##region of face is used to crop the image[rows:columns]
                roi_face = cv2.resize(roi_face, (200,200)) # used to resize the image in (200,200)
                roi_face_gray = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY) #face will be converted into gray.
                cv2.imwrite("FACE-DATASET/user." + str(count) +".jpg", roi_face_gray)
            cv2.imshow("MY DATA",image)
        if count > 50: #COUNT ONLY 50 IMAGES
            break
        if cv2.waitKey(1) == ord("q"):
            break
        
    cam.release() # TO STOP CAMERA
    cv2.destroyAllWindows() # TO DESTROY WINDOW
                
            


# In[12]:


Create_Dataset()


# In[13]:


from os import listdir ##give you the list inside the directoy
def Train_Model():
    path = "FACE-DATASET/"
    all_images = listdir(path)
    
    train_data = []  #independent data 
    labels = []
    for img in all_images:
        print(path + img)
        


# In[14]:


Train_Model()


# In[15]:


from os import listdir
def Train_Model():
    path = "Face-DataSet/"
    all_images = listdir(path)
    
    train_data = []
    labels = []
    for img in all_images:
        if img[-3:] != "jpg":
            continue
        image_path = (path + img)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_array = np.asarray(image, dtype = np.uint8)
        train_data.append(image_array)
        labels.append(int(img.split(".")[-2]))
    labels = np.array(labels)   
    face_model = cv2.face_LBPHFaceRecognizer.create()
    face_model.train(train_data, labels)
    
    ## Save Our Trained Model
    face_model.save("trained_Model.yml")
    print("We saved trained Model As *trained_Model.yml* name") 


# In[16]:


Train_Model()


# In[ ]:





# In[ ]:





# In[17]:


def DetectMyFace():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") ##classifier which is used to detect particular objects from the source.
    cam = cv2.VideoCapture(0)
    model = cv2.face_LBPHFaceRecognizer.create() ## Local Binary Patterns Histograms(face recognization algo)
    model.read("trained_Model.yml")
    while True:
        ret,image = cam.read()
        if ret == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,10)
            for x,y,w,h in faces:
                roi_face = image[y:y+h, x:x+w] #used to crop image(REGION OF INTEREST)
                roi_face = cv2.resize(roi_face, (200,200))
                roi_face_gray = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY)
                pred = model.predict(roi_face_gray)
                print(pred)
                if pred[1] < 40:
                        cv2.putText(image, "HEY DEEPANKAR!!", (x,y-15),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2) #it will draw the reactangle if image distance is less than 40.
                else:
                    cv2.putText(image, "WHO ARE YOU", (x,y-15),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
                     
        cv2.imshow("Face Detection", image)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
    


# In[18]:


DetectMyFace()


# In[ ]:




