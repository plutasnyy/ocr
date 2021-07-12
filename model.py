import cv2 as cv
import imutils as imutils
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from imutils.contours import sort_contours
from torchvision import transforms


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Linear(800, 10)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def predict_image(self, image):
        gray_img = cv.cvtColor(np.array(image.convert('L')), cv.COLOR_BGR2RGB)
        _, binary_img = cv.threshold(gray_img, 127, 255, 0)
        binary_img = cv.cvtColor(binary_img, cv.COLOR_RGB2GRAY)

        contours = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours, bounding_boxes = sort_contours(contours, method="left-to-right")

        digits = []
        for i, (c, box) in enumerate(zip(contours, bounding_boxes)):
            (x, y, w, h) = box
            if w < 7 or h < 7:  # I am a bit afraid about 1, which can be very thick
                continue

            digit = binary_img[y:y + h, x:x + w]

            target = max(digit.shape)
            diff_left_right = (target - digit.shape[1]) // 2
            diff_top_bottom = (target - digit.shape[0]) // 2

            padded_image = cv.copyMakeBorder(digit, top=diff_top_bottom, bottom=diff_top_bottom, left=diff_left_right,
                                             right=diff_left_right, borderType=cv.BORDER_CONSTANT, value=0)
            resized_image = imutils.resize(padded_image, width=22, height=22, inter=cv.INTER_AREA)
            padded_to_28_image = cv.copyMakeBorder(resized_image, top=3, bottom=3, left=3, right=3,
                                                   borderType=cv.BORDER_CONSTANT, value=0)

            transformed_image = self.transform(padded_to_28_image / 255).unsqueeze(0).float()
            y_pred = self(transformed_image).argmax(dim=1).item()
            digits.append(y_pred)

        return ''.join(map(str, digits))
