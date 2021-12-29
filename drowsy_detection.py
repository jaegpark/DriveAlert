import cv2
import os
import torch
import CNN
from pygame import mixer
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable as V



state_dict = torch.load('model_eyes.pt')
#print(state_dict.keys())

Model = CNN.Net()

Model.load_state_dict(state_dict)
Model.eval()

for param in Model.parameters():
    param.requires_grad = False

#print(Model)


mixer.init()
sound = mixer.Sound('alarm.wav')

# Initialize xml directories for face/eyes haar CascadeClassifier
face_classifier = cv2.CascadeClassifier('haar_cascade_files\haarcascade_frontalface_alt.xml')
lefteye_classifier = cv2.CascadeClassifier('haar_cascade_files\haarcascade_lefteye_2splits.xml')
righteye_classifer = cv2.CascadeClassifier('haar_cascade_files\haarcascade_righteye_2splits.xml')


labels=['Close','Open']

path = os.getcwd()

# Get video feed 
capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc=2
rpred = None
lpred = None

while(True):    # Infinite loop to continually get updates from webcam.
    ret, frame = capture.read()     # get frame data and store as img (frame is an image type)
    if not ret:
        break
    height,width = frame.shape[:2] 

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert cam data to RGB! 
    img = transforms.ToTensor()(img) # now looks like C X H X W
    #print(img.shape)            # img : tensor : 3 x 480 x 640 (C X H X W)
    #print(frame.shape)          # frame: image : H X W X C

    # Detect faces/eyes ROI 
    faces = face_classifier.detectMultiScale(frame, minNeighbors = 5, scaleFactor = 1.1, minSize = (25,25))
    left_eye = lefteye_classifier.detectMultiScale(frame)
    right_eye =  righteye_classifer.detectMultiScale(frame)

    eyetrans = transforms.Compose([transforms.Resize(24), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                                    
    cv2.rectangle(frame, (0, height - 50) , (200, height) , (0, 0, 0) , thickness=cv2.FILLED )

    # Draw rectangle around face ROI
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y) , (x+w, y+h) , (100, 100, 100) , 1 )

    # Draw rectangle around right eye and classify eye as open/closed
    for (x, y, w, h) in right_eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        r_eye = frame[y:y+h, x:x+w]     
        guh = Image.fromarray(r_eye)
        guh.save('test_righteye.jpg')
        #print(r_eye.shape)          # r_eye is an IMG H X W X C
        r_eye = eyetrans(Image.fromarray(r_eye)) 
        #print(r_eye.shape)          # r_eye is NOW an 
        
        rpred = Model.forward(V(r_eye.unsqueeze(0)))
        ps = torch.exp(rpred)
        #print(ps)
        top_p, top_class = ps.topk(1, dim=1)    # first index is closed true, second is open true
        r_labels = 'closed' if top_class == 0 else 'open'
        print('Right eye: ', r_labels)
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        l_eye = frame[y:y+h, x:x+w]     
        guh = Image.fromarray(l_eye)
        guh.save('test_lefteye.jpg')
        #print(l_eye.shape)          # l_eye is an IMG H X W X C
        l_eye = eyetrans(Image.fromarray(l_eye)) 
        #print(l_eye.shape)          # l_eye is NOW a PIL image
        
        lpred = Model.forward(V(l_eye.unsqueeze(0)))
        ps = torch.exp(lpred)   # probabilities are model exponentized
        #print(ps)
        top_p, top_class = ps.topk(1, dim=1)    # first index is closed true, second is open true
        l_labels = 'closed' if top_class == 0 else 'open'
        print('Left eye: ', l_labels)
        break
    
    if(l_labels == 'closed' and r_labels == 'closed'):
        score += 1
        cv2.putText(frame, "Eyes Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Eyes Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Reset score to 0 if eyes remain open for long periods of time
    if(score < 0):
        score = 0

    # Display score
    cv2.putText(frame, 'Drowsy Score:'+str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Alarm control
    if(score > 15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
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