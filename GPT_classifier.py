import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 25 * 25, 512)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 25 * 25)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



import torch.optim as optim

############# Code by human #############
from torchvision.transforms import transforms
from imagedataset import ImageDataset
from torch.utils.data import DataLoader, random_split

num_epochs = 2
batch_size = 32
norm_transform = transforms.Normalize(
    (132.3501, 127.2977, 131.0638),
    (55.5031, 62.3274, 64.1869)
)

dataset = ImageDataset()
train_size = int(0.75*len(dataset))
test_size  = len(dataset) - train_size
train, test = random_split(dataset, [train_size, test_size])
trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test, batch_size=batch_size, shuffle=False)
#########################################

# Define the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model on your dataset
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 25 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


torch.save(model.state_dict(), 'model.pth')


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, testloader):
    # Set model to evaluation mode
    model.eval()

    # Initialize variables for tracking accuracy and classwise accuracy
    total = 0
    correct = 0
    class_correct = list(0. for i in range(29))
    class_total = list(0. for i in range(29))

    # Iterate over the test set and predict labels
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute classwise accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Compute overall accuracy
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))

    # Compute and plot classwise accuracy
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(29)]
    class_names = [str(i) for i in range(29)]
    y_pos = np.arange(len(class_names))
    plt.bar(y_pos, class_accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, class_names)
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Classwise accuracy of the network on the test images')
    plt.show()

# Assuming you have trained your model and saved it to a file named 'model.pth'
model = CNN()
model.load_state_dict(torch.load('model.pth'))

# Test the model and plot classwise accuracy
test_model(model, testloader)