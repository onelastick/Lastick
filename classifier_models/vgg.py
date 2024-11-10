import torch
import torch.nn as nn

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self, num_classes=10):
        """
        Constructor
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        # self.block_size = [2, 2, 3, 3, 3]
        # self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        # self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        # self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        # self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        # self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        # self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        # self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, x):
        """ Pytorch forward

        Args:
            x: input image (32x32)

        Returns: class logits

        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # x = F.relu(self.conv_1_1(x))
        # x = F.relu(self.conv_1_2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv_2_1(x))
        # x = F.relu(self.conv_2_2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv_3_1(x))
        # x = F.relu(self.conv_3_2(x))
        # x = F.relu(self.conv_3_3(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv_4_1(x))
        # x = F.relu(self.conv_4_2(x))
        # x = F.relu(self.conv_4_3(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv_5_1(x))
        # x = F.relu(self.conv_5_2(x))
        # x = F.relu(self.conv_5_3(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc6(x))
        # x = F.dropout(x, 0.5, self.training)
        # x = F.relu(self.fc7(x))
        # x = F.dropout(x, 0.5, self.training)
        # return self.fc8(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        # 8 Convolutional layers
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # 3 Max-Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # 3 Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        # Flatten layer
        self.flatten = nn.Flatten()

        # 2 Fully connected (Dense) layers
        self.fc1 = nn.Linear(512 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return x