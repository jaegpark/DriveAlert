import torch
import numpy as np


# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

#####################################################################################################
#                                        START OF DATA LOADING                                      #
#####################################################################################################
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

batch_size = 20
num_workers = 0
valid_size = 0.2

train_trans = transforms.Compose([
                transforms.Resize((24, 24)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

train_dir = 'data/train'
test_dir = 'data/test'

train_data = datasets.ImageFolder(train_dir, transform=train_trans)
test_data = datasets.ImageFolder(test_dir, transform=train_trans)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]


# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

classes = ['Closed','Open']

print(train_loader)
#####################################################################################################
#                                         END OF DATA LOADING                                       #
#####################################################################################################


import matplotlib.pyplot as plt
#%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(10):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
plt.show()


#####################################################################################################
#                                   DEFINING THE NEURAL NETWORK                                     #
#####################################################################################################

import torch.nn as nn
import torch.nn.functional as F

# define CNN Architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv layer
        # sees 24x24 x3 (RGB)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # in depth = 3, out depth = 16, ksize = 3, padding 1, stride 1 (default)
        
        # sees 12x12 x16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # in depth = 16, out depth = 32, ksize = 3, padding 1, stride 1

        # sees 6x6 x32
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # final out of conv: 3x3 x64

        self.fc1 = nn.Linear(3*3*64, 100, bias=True)
        self.fc2 = nn.Linear(100, 2, bias=True)
        self.dropout = nn.Dropout(p=0.25)

        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 3*3*64)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

model = Net()
print(model)

if train_on_gpu:
    model.cuda()

# loss func and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
#####################################################################################################
#                                   END OF NEURAL NETWORK CLASS                                     #
#####################################################################################################



#####################################################################################################
#                                        START OF TRAINING                                          #
#####################################################################################################

num_epochs = 10
valid_loss_min = np.Inf

for epoch in range(1, num_epochs+1):
    train_loss = 0
    valid_loss = 0

    ##### TRANING STEP #####
    model.train()       # switch model modes
    for data, labels in train_loader:
        # move tensors to GPU if available
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        # clear grads
        optimizer.zero_grad()

        model_pred = model(data)
        loss = criterion(model_pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    ##### VALIDATION STEP #####
    model.eval()
    for data, labels, in valid_loader:
        # move tensors to GPU if available
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        output = model(data)
        loss = criterion(output, labels)
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loss.dataset)

    print('Epoch: {} \nTraining Loss: {:.5f} \nValidation Loss: {:.5f}'.format(epoch, train_loss, valid_loss))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model_eyes.pt')
        valid_loss_min = valid_loss
