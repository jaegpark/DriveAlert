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
#plt.show()
plt.close()


#####################################################################################################
#                                   DEFINING THE NEURAL NETWORK                                     #
#####################################################################################################

import torch.nn as nn
import CNN

model = CNN.Net()
#print(model)

if train_on_gpu:
    model.cuda()

# loss func and optimizer
import torch.optim as optim

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#####################################################################################################
#                                        START OF TRAINING                                          #
#####################################################################################################

num_epochs = 30
valid_loss_min = np.Inf

valid_losses = []
train_losses = []
test_losses = []
accuracies = []
for epoch in range(1, num_epochs+1):
    train_loss = 0
    valid_loss = 0
    test_loss = 0

    ##### TRANING STEP #####
    model.train()       # switch model modes
    for data, labels in train_loader:
        # move tensors to GPU if available
        if train_on_gpu:
            data, labels = data.cuda(), labels.cuda()
        # clear grads
        optimizer.zero_grad()

        model_pred = model.forward(data)
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
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    with torch.no_grad():
        test_loss = 0
        accuracy = 0
        model.eval()
        for data, labels in test_loader:
            if train_on_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            log_ps = model.forward(data)
            loss = criterion(log_ps, labels)
            test_loss += loss

            ps = torch.exp(log_ps)   # since models output is log soft max, need to exp to get P
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()

    test_losses.append(test_loss/len(test_loader))
    accuracies.append(accuracy/len(test_loader))

    print('\nEpoch: {} \nTraining Loss: {:.5f} \nValidation Loss: {:.5f}\nTest Loss: {:.5f}\nAccuracy: {:.3f}'.format(epoch,
     train_loss, valid_loss, test_loss, accuracies[-1]))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'model_eyes.pt')
        valid_loss_min = valid_loss

# GRAPH Training loss 
f1 = plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend(frameon=False)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.savefig('training_vs_validation_loss.png')

f2 = plt.figure()
plt.plot(accuracies, label='Accuracy')
plt.legend(frameon=False)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('training_accuracy.png')