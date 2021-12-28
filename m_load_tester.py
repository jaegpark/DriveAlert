import CNN
import torch
import numpy as np

state_dict = torch.load('model_eyes.pt')
#print(state_dict.keys())

Model = CNN.Net()

Model.load_state_dict(state_dict)

print(Model)

