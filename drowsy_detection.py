import cv2
import os
import numpy as np
from pygame import mixer
import time
import torch
import CNN


state_dict = torch.load('model_eyes.pt')
#print(state_dict.keys())

Model = CNN.Net()

Model.load_state_dict(state_dict)

print(Model)



# Initialize alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Initialize xml directories for face/eyes haar CascadeClassifier
face_classifier = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
lefteye_classifier = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
righteye_classifer = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


labels=['Close','Open']


path = os.getcwd()

# Get video feed 
capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]


### DETECTION LOOP

while(True):    # Infinite loop to continually get updates from webcam.
    ret, frame = capture.read()     # get frame data and store as img (frame is an image type)
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert cam data to grayscale
    
    # Detect faces/eyes ROI 
    faces = face_classifier.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = lefteye_classifier.detectMultiScale(gray)
    right_eye =  righteye_classifer.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    # Draw rectangle around face ROI
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    # Draw rectangle around right eye and classify eye as open/closed
    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        #rpred = model.predict(r_eye)
        rpred = (Model.predict(r_eye) > 0.5)#.astype("int")
        if(rpred[0][0] == 0):
            labels='Open'
            #print('right eye open')
        if(rpred[0][0] == 1):
            labels='Closed'
            #print('right eye closed')
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        #lpred = model.predict(l_eye)
        lpred = (Model.predict(l_eye) > 0.5)

        if(lpred[0][0]==0):
            labels='Open'  
            print('left eye open') 
        if(lpred[0][0]==1):
            labels='Closed'
            print('left eye closed')
        break

    if(rpred[0][0]==1 and lpred[0][0]==1):
        score += 1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score -= 1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    # Reset score to 0 if eyes remain open for long periods of time
    if(score<0):
        score = 0

    # Display score
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    # Alarm control
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
