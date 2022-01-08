import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 11, 4, 0)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(48, 128, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # 256px, 61 - зависимость от изображения и MaxPool2d (при 32px - 5)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 4)

        print('Model created!')

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        # print(x.size())

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        # print(x.size())

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        # print(x.size())

        x = x.view(-1, 128 * 6 * 6)  # 256px, 61 - зависимость от изображения MaxPool2d (при 32px - 5)
        # print(x.size())

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.size())

        return x
