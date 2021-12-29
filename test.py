import torch
import CNN
import numpy as np
import cv2
from torchvision import transforms
from torch.autograd import Variable as V
from PIL import Image


state_dict = torch.load('model_eyes.pt')
#print(state_dict.keys())

Model = CNN.Net()
Model.load_state_dict(state_dict)
Model.eval()

for param in Model.parameters():
    param.requires_grad = False

for param in Model.parameters():
    param.requires_grad = False
eyetrans = transforms.Compose([transforms.Resize(24), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
img = np.asarray(Image.open('test.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

reye = eyetrans(Image.fromarray(img))
rpred = Model.forward(V(reye.unsqueeze(0)))
ps = torch.exp(rpred)
print(ps)
top_p, top_class = ps.topk(1, dim=1)
print('closed' if top_class == 0 else 'open')


'''
## WEBCAM TEST:

capture = cv2.VideoCapture(0)
ret, frame = capture.read()     # get frame data and store as img (frame is an image type)
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert cam data to RGB! 
img = transforms.ToTensor()(img) # now looks like C X H X W
#print(img.shape)            # img : tensor : 3 x 480 x 640 (C X H X W)
#print(frame.shape)          # frame: image : H X W X C
righteye_classifer = cv2.CascadeClassifier('haar_cascade_files\haarcascade_righteye_2splits.xml')
right_eye =  righteye_classifer.detectMultiScale(frame)



for (x, y, w, h) in right_eye:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
    r_eye = frame[y:y+h, x:x+w]     
    #print(r_eye.shape)          # r_eye is an IMG H X W X C
    r_eye = eyetrans(Image.fromarray(r_eye)) 
    rpred = Model.forward(V(r_eye.unsqueeze(0)))
    ps = torch.exp(rpred)
    print(ps)
    top_p, top_class = ps.topk(1, dim=1)    # first index is open true, second is closed true
    print(top_class)
    break
capture.release()
cv2.destroyAllWindows()
'''